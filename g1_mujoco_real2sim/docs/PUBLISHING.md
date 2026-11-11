# Publishing (maintainers)

How to put the repository on GitHub and the datasets on the Hugging Face Hub. Nothing here runs by itself: every upload is a command you type. Back to the [main README](../README.md).

## 1. Before the first push: checklist

No secrets in the tree. Only variable names such as `DREAM_API_KEY` may show up, never values:

```bash
grep -rIn -i -E "api[_-]?key|token|hf_[A-Za-z0-9]{20,}|dl_live" --exclude-dir=third_party --exclude-dir=.git .
```

No personal paths. This should print nothing:

```bash
grep -rIn "/home/" --exclude-dir=third_party --exclude-dir=.git .
```

Tests pass:

```bash
source scripts/env.sh g1lab
python -m pytest 04_recording 05_training -q
```

| Also check | How |
|---|---|
| Data is not in git | `datasets/`, `outputs/`, `third_party/`, `03_real2sim/calib/real_refs/` are in [.gitignore](../.gitignore) |
| Media files exist | Every file referenced in the READMEs is in `docs/media/` (list: [media/TODO.md](media/TODO.md)) |
| `<LAB_HF_ORG>` is replaced | Section 2 |
| The open decisions are settled | Section 6 |

## 2. Replace `<LAB_HF_ORG>`

The Hugging Face organisation is written as the placeholder `<LAB_HF_ORG>` in the docs and dataset cards. Once the name is decided (example below: `mobilab-udesc`), replace it everywhere with one command, from the repository root:

```bash
grep -rIl "<LAB_HF_ORG>" --exclude-dir=third_party --exclude-dir=.git . | xargs sed -i 's/<LAB_HF_ORG>/mobilab-udesc/g'
```

Check that none is left:

```bash
grep -rIn "<LAB_HF_ORG>" --exclude-dir=third_party --exclude-dir=.git .
```

You should see no output.

Notes:

- `04_recording/hf/upload_datasets.sh` takes the organisation as its first argument and fills it into the cards at upload time. It also contains the placeholder as a default value; the command above replaces that too, which is fine.
- This section then shows your organisation name in the commands. That is fine as well.
- In commands the placeholder is written in quotes (`"<LAB_HF_ORG>/..."`), because a bare `<` is a shell redirect. The command above matches the quoted form too; the quotes stay and do no harm.

## 3. Push the repository to GitHub

The target is <https://github.com/MOBILAB-UDESC/g1-mujoco-real2sim>. Create the empty repository in the organisation first (web page, or `gh repo create MOBILAB-UDESC/g1-mujoco-real2sim --private`).

The folder is already a git repository: branch `main`, one commit per stage, tags `stage-1` ... `stage-5`. Review it first:

```bash
git status
git log --oneline
```

- `git status` must be clean: commit what is still listed.
- It must not list `third_party/`, `datasets/`, `outputs/` or large files.

Then add the remote and push the branch with its tags:

```bash
git remote add origin https://github.com/MOBILAB-UDESC/g1-mujoco-real2sim.git
git push -u origin main --tags
```

Then open the repository page and check that the hero GIF, the stage thumbnails and the Mermaid diagrams render.

- Keep the repository private until the dataset license and the media are confirmed, then switch it to public in the GitHub settings.
- GIFs larger than about 10 MB load slowly on the README. Keep `docs/media/hero.gif` small.

## 4. Upload the datasets to the Hugging Face Hub

Log in once with a token that has write access to the organisation:

```bash
source scripts/env.sh g1lab
hf auth login
```

Put the datasets where the script looks for them (or set the variables in the table):

| Variable | Default folder | Hub name | robot_type |
|---|---|---|---|
| `REAL_DIR` | `datasets/pick_cube_real` | `<LAB_HF_ORG>/g1_pick_cube_real` | `Unitree_G1_ArmsOnly_3Cam` |
| `REAL_WAIST_DIR` | `datasets/real_waist` | `<LAB_HF_ORG>/g1_pick_cube_real_waist` | `Unitree_G1_ArmsWaist_3Cam` |
| `SIM_DIR` | `datasets/sim_scripted_waist` | `<LAB_HF_ORG>/g1_box_into_box_sim_scripted` | `Unitree_G1_ArmsWaist_3Cam` |

Print the commands:

```bash
04_recording/hf/upload_datasets.sh "<LAB_HF_ORG>"
```

You should see, per dataset, a summary line (episodes, frames, robot_type, size) and four commands:

```
hf repo create "<LAB_HF_ORG>/g1_pick_cube_real" --repo-type dataset --private
hf upload "<LAB_HF_ORG>/g1_pick_cube_real" .../datasets/pick_cube_real . --repo-type dataset --exclude "images/*" --commit-message ...
hf upload "<LAB_HF_ORG>/g1_pick_cube_real" .../g1_hf_cards/g1_pick_cube_real/README.md README.md --repo-type dataset --commit-message "Add dataset card"
hf repo tag create "<LAB_HF_ORG>/g1_pick_cube_real" v3.0 --repo-type dataset
```

Review them, then copy and run them one by one.

- **The script uploads nothing.** It only prints.
- A dataset whose folder is missing is reported as `SKIPPED`.
- The `v3.0` tag is required: LeRobot refuses to load a Hub dataset without its format-version tag.
- The cards are in [04_recording/hf/cards](../04_recording/hf/cards). Their license line says `apache-2.0` with a TODO: confirm the data license with the lab before making a dataset public.
- Repos are created private. Make them public on the Hub page when ready.

Check a published dataset (this downloads it):

```bash
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; print(D('<LAB_HF_ORG>/g1_pick_cube_real_waist'))"
```

## 5. After publishing

| Task | Where |
|---|---|
| Add dataset links to the main README table | [README.md](../README.md), section 7 |
| Update the status table when something is tested (headset wrist views, real-robot run, unitree_bridge) | [README.md](../README.md), section 8 |
| Take the remaining pictures and clips | [media/TODO.md](media/TODO.md) |
| Re-render the pictures and GIFs after a scene change | `python docs/make_media.py` (see its docstring; `--log` adds the episode GIFs) |

## 6. Decisions for the maintainer

Open points that only a maintainer can settle. Go through them before the repository or a dataset becomes public.

| Decision | State now | Where |
|---|---|---|
| Hugging Face organisation name | Placeholder `<LAB_HF_ORG>` | Section 2 |
| Dataset license | The cards say `apache-2.0 # TODO` | [04_recording/hf/cards](../04_recording/hf/cards) |
| Task string of the real dataset | It is `pick up cube.` (xr_teleoperate's default), although the task is box-into-box. Rename it or explain it on the card. | [g1_pick_cube_real card](../04_recording/hf/cards/g1_pick_cube_real/README.md) |
| Git author of the commits | `joao.girardi@fablabjoinville.com.br`. Change it with a rebase if another identity is wanted. | `git log` |
| Lab photos | People were blurred in `2.jpg`, `4.jpg`, `room/8.jpg`, `room/9.jpg`: confirm it is enough. Other logos visible: Creality, Clearpath. | [03_real2sim/assets/lab/ref](../03_real2sim/assets/lab/ref) |
| Pin of unitree_lerobot | `UNITREE_LEROBOT_COMMIT` is `main`. Pin it to a commit. | [install/install.sh](../install/install.sh) |
| License of the bracket CAD | [MOBILAB-UDESC/cad-models](https://github.com/MOBILAB-UDESC/cad-models) has no license file. | [THIRD_PARTY.md](../THIRD_PARTY.md), section 3 |
| Redistributed vendor meshes | Unitree's 5010 wrist and rubber-hand STLs (unitree_ros, BSD-3-Clause) and Intel's D405 mesh are stored in this repository. Needs a maintainer's sign-off. | [THIRD_PARTY.md](../THIRD_PARTY.md), section 3 |
