"""
Run ONE episode with a trained checkpoint, log everything the dashboard
needs, and bake it into a standalone HTML file.

Usage:

    python -m Marl.mappo.dashboard_log \
        --checkpoint checkpoints/attention_test_curriculum/mappo_final.pt \
        --red-agent finite --seed 123

Optional:

    --freeze-trust
    --ablation abl.json
"""

import argparse
import json
import os

import numpy as np
import torch

from .env import CC4Env
from .mappo import MAPPO
from .communication.evaluator import MessageEvaluator
from .communication.schema import HostStatus

from .train import (
    pad_observation,
    episode_is_done,
    get_current_communication_for_agent,
    get_receiver_relevance,
)

from .action_mask import compute_padded_mask, explain_mask
from .evaluate import get_checkpoint_num_targets, RED_AGENTS
from .config import NUM_AGENTS, OBS_DIM, EPISODE_LENGTH


ZONES = [
    "RZ-A",
    "OZ-A",
    "RZ-B",
    "OZ-B",
    "HQ",
]


def f(x):
    return None if x is None else float(x)


def zone_counts(gt):
    """Return compromised and suspicious host counts."""

    if gt is None:
        return 0, 0

    st = [
        v["status"]
        for v in gt["target_status"].values()
    ]

    return (
        sum(s == HostStatus.COMPROMISED for s in st),
        sum(s == HostStatus.SUSPICIOUS for s in st),
    )


@torch.no_grad()
def run(args):

    n_host, n_sub = get_checkpoint_num_targets(
        args.checkpoint
    )

    env0 = CC4Env(
        red_agent_class=RED_AGENTS[args.red_agent]
    )

    n_host = n_host or env0.get_num_targets()
    n_sub = n_sub or env0.get_num_subnet_targets()

    ppo = MAPPO(
        num_host_targets=n_host,
        num_subnet_targets=n_sub,
    )

    ppo.load(args.checkpoint)
    ppo.eval()

    evaluator = MessageEvaluator()

    env = CC4Env(
        red_agent_class=RED_AGENTS[args.red_agent]
    )

    names = sorted(env.possible_agents)

    dims = env.get_observation_dims()

    obs_dict, _ = env.reset(
        seed=args.seed
    )

    prev_msgs = None

    steps = []

    for t in range(EPISODE_LENGTH):

        obs = np.zeros(
            (NUM_AGENTS, OBS_DIM),
            dtype=np.float32,
        )

        for i, n in enumerate(names):

            obs[i] = pad_observation(
                obs_dict[n],
                dims[n],
            )

        host_masks = (
            env.get_all_host_active_masks()
        )

        global_obs = obs.reshape(-1)

        actions = {}
        att_rows = []
        mask_info = []

        # ---------------------------------------------------------
        # Select actions
        # ---------------------------------------------------------

        for i, n in enumerate(names):

            _, reasons = explain_mask(
                env,
                n,
            )

            mask = compute_padded_mask(
                env,
                n,
            )

            mask_info.append(
                {
                    "enabled": int(mask.sum()),
                    "suppressed": len(reasons),
                    "reasons": list(reasons.values())[:8],
                }
            )

            rm, tw = (
                get_current_communication_for_agent(
                    ppo,
                    i,
                    prev_msgs,
                )
            )

            action = ppo.select_action(
                observation=obs[i],
                action_mask=mask,
                global_state=global_obs,
                agent_id=i,
                received_messages=rm,
                trust_weights=tw,
                host_active_mask=host_masks[i],
            )[0]

            actions[n] = int(action)

            att = getattr(
                ppo.actor,
                "last_communication_attention",
                None,
            )

            row = None

            if rm is not None and att is not None:

                cand = att.reshape(-1).tolist()

                if len(cand) == NUM_AGENTS:
                    row = [
                        float(v)
                        for v in cand
                    ]

            att_rows.append(row)

        # ---------------------------------------------------------
        # Generate outgoing messages
        # ---------------------------------------------------------

        (
            vecs,
            structured,
            _,
            _,
            _,
        ) = ppo.get_outgoing_messages(
            obs,
            return_decoded=True,
            host_active_mask=host_masks,
        )

        # ---------------------------------------------------------
        # Environment step
        # ---------------------------------------------------------

        obs_dict, rewards, term, trunc, _ = (
            env.step(actions)
        )

        done = episode_is_done(
            term,
            trunc,
        )

        # ---------------------------------------------------------
        # Ground truth + message grading
        # ---------------------------------------------------------

        comp = []
        susp = []
        msgs = []

        for s in range(NUM_AGENTS):

            gt = env.get_ground_truth(s)

            c, u = zone_counts(gt)

            comp.append(c)
            susp.append(u)

            m = structured[s]

            d = m.as_dict()

            d.update(
                sender=s,
                empty=bool(m.is_empty()),
                q=None,
                sc=None,
            )

            if gt is not None and not d["empty"]:

                ev = evaluator.evaluate(
                    m,
                    gt,
                    receiver_relevance=1.0,
                )

                d["q"] = f(
                    evaluator.confidence_adjusted_score(
                        ev
                    )
                )

                d["sc"] = {
                    "event": f(ev.event_score),
                    "target": f(ev.target_score),
                    "threat": f(ev.threat_score),
                    "status": f(ev.status_score),
                }

                # -------------------------------------------------
                # Update trust for every receiver
                # -------------------------------------------------

                if not args.freeze_trust:

                    for r in range(NUM_AGENTS):

                        if r == s:
                            continue

                        evr = evaluator.evaluate(
                            m,
                            gt,
                            receiver_relevance=(
                                get_receiver_relevance(
                                    s,
                                    r,
                                )
                            ),
                        )

                        ppo.update_trust(
                            sender=s,
                            receiver=r,
                            message_quality=(
                                evaluator.confidence_adjusted_score(
                                    evr
                                )
                            ),
                        )

            msgs.append(d)

        # ---------------------------------------------------------
        # Trust matrix
        # ---------------------------------------------------------

        trust = [
            [
                0.0
                if s == r
                else float(
                    ppo.get_trust(s, r)
                )
                for r in range(NUM_AGENTS)
            ]
            for s in range(NUM_AGENTS)
        ]

        # ---------------------------------------------------------
        # Save this step
        # ---------------------------------------------------------

        steps.append(
            {
                "t": t,
                "phase": int(obs[0][0]),
                "comp": comp,
                "susp": susp,
                "reward": [
                    float(rewards[n])
                    for n in names
                ],
                "actions": [
                    env.action_labels(n)[
                        actions[n]
                    ]
                    if actions[n]
                    < len(env.action_labels(n))
                    else "pad"
                    for n in names
                ],
                "msgs": msgs,
                "trust": trust,
                "att": att_rows,
                "mask": mask_info,
            }
        )

        prev_msgs = vecs.detach()

        if done:
            break

    # -------------------------------------------------------------
    # Optional ablation data
    # -------------------------------------------------------------

    ablation = None

    if (
        args.ablation
        and os.path.isfile(args.ablation)
    ):

        ablation = json.load(
            open(args.ablation)
        )

    return {
        "meta": {
            "checkpoint": os.path.basename(
                args.checkpoint
            ),
            "red_agent": args.red_agent,
            "seed": args.seed,
            "zones": ZONES,
            "trust_frozen": args.freeze_trust,
        },
        "steps": steps,
        "ablation": ablation,
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--checkpoint",
        required=True,
    )

    ap.add_argument(
        "--red-agent",
        choices=[
            "random",
            "finite",
        ],
        default="finite",
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=123,
    )

    ap.add_argument(
        "--freeze-trust",
        action="store_true",
    )

    ap.add_argument(
        "--ablation",
        default=None,
    )

    ap.add_argument(
        "--template",
        default=os.path.join(
            os.path.dirname(__file__),
            "dashboard_template.html",
        ),
    )

    ap.add_argument(
        "--out",
        default="evaluation/dashboard",
    )

    args = ap.parse_args()

    # -------------------------------------------------------------
    # Run episode
    # -------------------------------------------------------------

    data = run(args)

    os.makedirs(
        args.out,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Save episode data
    # -------------------------------------------------------------

    json.dump(
        data,
        open(
            os.path.join(
                args.out,
                "episode_log.json",
            ),
            "w",
        ),
    )

    # -------------------------------------------------------------
    # Inject data into HTML template
    # -------------------------------------------------------------

    payload = json.dumps(
        data
    ).replace(
        "</",
        "<\\/",
    )

    html = (
        open(
            args.template,
            encoding="utf-8",
        )
        .read()
        .replace(
            "const D=/*__DATA__*/null;",
            "const D=" + payload + ";",
        )
    )

    # -------------------------------------------------------------
    # Write dashboard
    # -------------------------------------------------------------

    out_html = os.path.join(
        args.out,
        "dashboard.html",
    )

    open(
        out_html,
        "w",
        encoding="utf-8",
    ).write(html)

    print(
        f"Wrote {out_html} "
        f"({len(data['steps'])} steps)"
    )


if __name__ == "__main__":
    main()
