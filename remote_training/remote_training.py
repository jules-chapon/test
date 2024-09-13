"""Functions to initiate remote training"""

import kaggle
from pathlib import Path, PurePosixPath
import json

try:
    from remote_training.__kaggle_login import kaggle_users
except ImportError:
    raise ImportError(
        "Please create a __kaggle_login.py file with a kaggle_users"
        + "dict containing your Kaggle credentials.\n"
        + "See https://github.com/balthazarneveu/mva_pepites for more details."
    )
import argparse
import os
import sys
import subprocess
from src.configs import constants
from src.model.train import get_parser as get_train_parser
from typing import Optional


def get_git_branch_name():
    try:
        branch_name = (
            subprocess.check_output(["git", "branch", "--show-current"])
            .strip()
            .decode()
        )
        return branch_name
    except subprocess.CalledProcessError:
        return "Error: Could not determine the Git branch name."


def prepare_notebook(
    output_nb_path: Path,
    exp: int,
    branch: str,
    git_user: str = None,
    git_repo: str = None,
    template_nb_path: Path = os.path.join(
        constants.REMOTE_TRAINING_FOLDER, "remote_training.ipynb"
    ),
    output_dir: Path = constants.OUTPUT_FOLDER,
    dataset_files: Optional[list] = None,
):
    assert git_user is not None, "Please provide a git username for the repo"
    assert git_repo is not None, "Please provide a git repo name for the repo"
    expressions = [
        ("exp", f"{exp}"),
        ("branch", f"'{branch}'"),
        ("git_user", f"'{git_user}'"),
        ("git_repo", f"'{git_repo}'"),
        ("output_dir", "None" if output_dir is None else f"'{output_dir}'"),
        ("dataset_files", "None" if dataset_files is None else f"{dataset_files}"),
    ]
    with open(template_nb_path) as f:
        template_nb = f.readlines()
        for line_idx, li in enumerate(template_nb):
            for expr, expr_replace in expressions:
                if f"!!!{expr}!!!" in li:
                    template_nb[line_idx] = template_nb[line_idx].replace(
                        f"!!!{expr}!!!", expr_replace
                    )
        template_nb = "".join(template_nb)
    with open(output_nb_path, "w") as w:
        w.write(template_nb)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a model on Kaggle using a script\n"
        + "Help: https://github.com/balthazarneveu/mva_pepites"
    )
    parser.add_argument(
        "-n",
        "--notebook_id",
        type=str,
        help="Notebook name in kaggle",
        default=constants.NOTEBOOK_ID,
    )
    parser.add_argument(
        "-u", "--user", type=str, help="Kaggle user", choices=list(kaggle_users.keys())
    )
    parser.add_argument(
        "--branch", type=str, help="Git branch name", default=get_git_branch_name()
    )
    parser.add_argument("-p", "--push", action="store_true", help="Push")
    parser.add_argument(
        "-d", "--download", action="store_true", help="Download results"
    )
    get_train_parser(parser)
    args = parser.parse_args(argv)
    notebook_id = args.notebook_id
    exp_str = "_".join(f"{exp:04d}" for exp in args.exp)
    kaggle_user = kaggle_users[args.user]
    uname_kaggle = kaggle_user["username"]
    kaggle.api._load_config(kaggle_user)
    if args.download:
        tmp_dir = f"__tmp_{exp_str}"
        os.makedirs(tmp_dir, exist_ok=True)
        kaggle.api.kernels_output_cli(
            f"{kaggle_user['username']}/{notebook_id}", path=str(tmp_dir)
        )
        subprocess.run(["tar", "-xzf", tmp_dir / "output.tgz", constants.OUTPUT_FOLDER])
        # @FIXME: windows probably does not have tar command
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
        return
    kernel_root = f"__nb_{uname_kaggle}"
    kernel_path = os.path.join(kernel_root, exp_str)
    os.makedirs(kernel_path, exist_ok=True)
    branch = args.branch
    config = {
        "id": str(
            PurePosixPath(os.path.join(f"{kaggle_user['username']}"), notebook_id)
        ),
        "title": notebook_id.lower(),
        "code_file": f"{notebook_id}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true" if not args.cpu else "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": constants.KAGGLE_DATASET_LIST,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    prepare_notebook(
        os.path.join(kernel_path, notebook_id) + ".ipynb",
        args.exp,
        branch,
        git_user=constants.GIT_USER,
        git_repo=constants.GIT_REPO,
    )
    assert os.path.exists(os.path.join(kernel_path, notebook_id) + ".ipynb")
    with open(kernel_path / "kernel-metadata.json", "w") as f:
        json.dump(config, f, indent=4)

    if args.push:
        kaggle.api.kernels_push_cli(str(kernel_path))


if __name__ == "__main__":
    main(sys.argv[1:])
