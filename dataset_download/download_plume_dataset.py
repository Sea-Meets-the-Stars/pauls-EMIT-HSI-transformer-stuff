"""
download_dataset.py
-------------------
Entry point for building an EMIT hyperspectral plume dataset.

CLI usage
---------
    python download_dataset.py \\
        --gas_type       ch4 \\
        --output_dir     /data/emit_dataset \\
        --max_granules   50 \\
        --start_date     2023-01-01 \\
        --end_date       2025-12-31 \\
        --cloud_cover_max 3 \\
        --cube_format    npy \\
        --overwrite

Notebook / interactive usage
----------------------------
    from download_dataset import (
        search_granule_ids_for_gas_type,
        run_full_dataset_build,
        run_single_granule_test,
        run_plume_stats_survey,
        run_preview_for_granule,
        reconstruct_dataset_index_from_disk,
    )

    # Test a single function in isolation:
    granule_ids = search_granule_ids_for_gas_type(gas_type="ch4", max_granules=10)
    run_single_granule_test(granule_id=granule_ids[0], output_dir="/tmp/emit_test")

All heavy lifting lives in emit_utils.py.  This file only orchestrates calls to
those functions and exposes a clean CLI / importable API.
"""

import argparse
import sys
import os
import shutil

import emit_utils

def reconstruct_dataset_index_from_disk(dataset_dir: str):
    """Delegates to :func:`emit_utils.reconstruct_dataset_index_from_disk`."""
    return emit_utils.reconstruct_dataset_index_from_disk(dataset_dir)


# ---------------------------------------------------------------------------
# Public API — importable by notebooks
# ---------------------------------------------------------------------------

def search_granule_ids_for_gas_type(
    gas_type: str = "ch4",
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
    max_granules: int = 100,
    cloud_cover_min: int = 0,
    cloud_cover_max: int = 3,
) -> list[str]:
    """
    Search NASA Earthdata for EMIT granule IDs containing plumes of the
    specified gas type within the given date range and cloud cover limits.

    Returns a list of granule ID strings (e.g. ['20230825T163454', ...]).
    """
    date_range = (start_date, end_date)
    cloud_cover_range = (cloud_cover_min, cloud_cover_max)

    granule_ids = emit_utils.get_plume_granule_ids(
        gas_type=gas_type,
        date_range=date_range,
        max_count=max_granules,
        cloud_cover_range=cloud_cover_range,
    )
    return granule_ids


def _granule_output_dir_candidates(dataset_dir: str, granule_id: str, gas_type: str) -> tuple[str, str]:
    """Same naming as emit_utils.save_one_granule_to_dataset (single- vs multi-plume)."""
    single = os.path.join(dataset_dir, f"{granule_id}_{gas_type}")
    multi = os.path.join(dataset_dir, f"{granule_id}_{gas_type}_multiple_plumes")
    return single, multi


def remove_incomplete_granule_dirs(dataset_dir: str, granule_ids: list[str], gas_type: str) -> None:
    """
    Delete granule output folders that exist but lack granule_metadata.json.
    The pipeline writes that file only after a full successful run; without it,
    save_one_granule_to_dataset would otherwise skip the granule as "already exists".
    """
    marker = "granule_metadata.json"
    for granule_id in granule_ids:
        for path in _granule_output_dir_candidates(dataset_dir, granule_id, gas_type):
            meta_path = os.path.join(path, marker)
            if os.path.isdir(path) and not os.path.isfile(meta_path):
                print(f"Removing incomplete granule directory (missing {marker}): {path}")
                shutil.rmtree(path, ignore_errors=True)
                

def run_full_dataset_build(
    granule_ids: list[str],
    output_dir: str,
    gas_type: str = "ch4",
    cube_format: str = "npy",
    overwrite: bool = False,
    run_mag1c: bool = True,
):
    """
    Download, chip, and save a complete labelled dataset for the supplied
    granule IDs.  Produces a dataset_index.csv in output_dir.

    Returns the dataset index as a pandas DataFrame.
    """
    dataset_index_df = emit_utils.build_dataset(
        granule_ids=granule_ids,
        dataset_dir=output_dir,
        cube_format=cube_format,
        overwrite=overwrite,
        gas_type=gas_type,
        run_mag1c=run_mag1c,
    )
    return dataset_index_df


def run_single_granule_test(
    granule_id: str,
    output_dir: str,
    gas_type: str = "ch4",
    cube_format: str = "npy",
    run_mag1c: bool = True,
):
    """
    Process exactly one granule end-to-end.  Useful for testing the pipeline
    in a notebook before committing to a full batch run.

    Returns the per-chip index DataFrame for that granule (or None if the
    granule was skipped / corrupted).
    """
    print(f"Running single-granule pipeline test for {granule_id} ({gas_type.upper()})...")
    single_granule_index_df = emit_utils.save_one_granule_to_dataset(
        granule_id=granule_id,
        dataset_dir=output_dir,
        gas_type=gas_type,
        cube_format=cube_format,
        overwrite=True,
        run_mag1c=run_mag1c,
    )
    return single_granule_index_df


def run_plume_stats_survey(
    gas_type: str = "ch4",
    max_granules: int = 100,
    include_difficulty_metrics: bool = True,
):
    """
    Survey plume statistics (and optionally difficulty metrics) across up to
    max_granules granules without building a full chipped dataset.  Useful for
    exploring the data distribution before committing to a large download.

    Returns a pandas DataFrame of per-plume statistics.
    """
    print(f"Surveying plume stats for {max_granules} {gas_type.upper()} granules...")
    plume_stats_df = emit_utils.survey_plume_stats(
        max_count=max_granules,
        gas_type=gas_type,
        include_difficulty_metrics=include_difficulty_metrics,
    )
    return plume_stats_df


def run_preview_for_granule(granule_id: str, gas_type: str = "ch4"):
    """
    Download and display browse images (L1B radiance + L2B enhancement) for a
    single granule.  Intended for interactive notebook exploration only — this
    calls matplotlib.pyplot.show() internally.
    """
    emit_utils.display_plume_previews(granule_id=granule_id, gas_type=gas_type)


def run_product_inventory_for_granule(granule_id: str) -> dict:
    """
    List every NASA EMIT product available for a granule ID, grouped by
    product type (L1B_RAD, L2B_CH4PLM, etc.).

    Returns a dict mapping product_type -> list of earthaccess result objects.
    """
    available_products_by_type = emit_utils.get_all_granule_products(granule_id)
    return available_products_by_type


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_plume_dataset.py",
        description=(
            "Search NASA Earthdata for EMIT plume granules and build a "
            "labelled hyperspectral dataset on disk."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Search parameters ---
    parser.add_argument(
        "--gas_type",
        type=str,
        default="ch4",
        choices=["ch4", "co2", "either", "both"],
        help="Gas type to search for.  'either' = union, 'both' = intersection.",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2023-01-01",
        metavar="YYYY-MM-DD",
        help="Start of the temporal search window.",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2025-12-31",
        metavar="YYYY-MM-DD",
        help="End of the temporal search window.",
    )
    parser.add_argument(
        "--max_granules",
        type=int,
        default=100,
        help="Maximum number of granules to search for and download.",
    )
    parser.add_argument(
        "--cloud_cover_min",
        type=int,
        default=0,
        help="Minimum cloud cover percentage filter (inclusive).",
    )
    parser.add_argument(
        "--cloud_cover_max",
        type=int,
        default=3,
        help="Maximum cloud cover percentage filter (inclusive).",
    )

    # --- Output / format parameters ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./emit_dataset",
        help="Root directory where the dataset will be written.",
    )
    parser.add_argument(
        "--cube_format",
        type=str,
        default="npy",
        choices=["npy", "hdf5"],
        help="On-disk format for hypercube chips.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-process granules that already exist in output_dir.",
    )
    parser.add_argument(
        "--reprocess_incomplete",
        action="store_true",
        default=False,
        help=(
            "Before building, remove granule folders under output_dir that have no "
            "granule_metadata.json (partial or interrupted runs) so they are rebuilt."
        ),
    )
    parser.add_argument(
        "--rebuild_index_only",
        action="store_true",
        default=False,
        help=(
            "Only scan output_dir and rewrite dataset_index.csv from on-disk metadata "
            "(no Earthdata search or downloads)."
        ),
    )

    # --- Execution modes ---
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--survey_only",
        action="store_true",
        default=False,
        help=(
            "Only run a plume statistics survey (no chipping / downloading of "
            "L1B hypercubes).  Prints a summary DataFrame and exits."
        ),
    )
    mode_group.add_argument(
        "--test_single_granule",
        type=str,
        default=None,
        metavar="GRANULE_ID",
        help=(
            "Run the full pipeline for exactly one granule ID and exit.  "
            "Useful for validating the environment before a large batch run."
        ),
    )
    parser.add_argument(
        "--no_mag1c",
        action="store_true",
        default=False,
        help="Skip MAG1C for CH4 granules (faster; CH4 mag1c.npy chips not written).",
    )
    return parser


def _run_from_cli(cli_args: argparse.Namespace):
    """Dispatch to the appropriate public function based on parsed CLI arguments."""

    # --- Mode: rebuild index from disk only ---
    if cli_args.rebuild_index_only:
        dataset_index_df = reconstruct_dataset_index_from_disk(cli_args.output_dir)
        print(f"\nDone. Dataset index shape: {dataset_index_df.shape}")
        return

    # --- Mode: single-granule test ---
    if cli_args.test_single_granule:
        single_granule_index_df = run_single_granule_test(
            granule_id=cli_args.test_single_granule,
            output_dir=cli_args.output_dir,
            gas_type=cli_args.gas_type,
            cube_format=cli_args.cube_format,
            run_mag1c=not cli_args.no_mag1c,
        )
        if single_granule_index_df is not None:
            print("\nSingle-granule index:")
            print(single_granule_index_df.to_string(index=False))
        reconstruct_dataset_index_from_disk(cli_args.output_dir)
        return

    # --- Mode: survey only (no hypercube download) ---
    if cli_args.survey_only:
        plume_stats_df = run_plume_stats_survey(
            gas_type=cli_args.gas_type,
            max_granules=cli_args.max_granules,
            include_difficulty_metrics=True,
        )
        print("\nPlume statistics survey results:")
        print(plume_stats_df.to_string(index=False))
        return

    # --- Mode: full dataset build (default) ---
    print("Step 1/2 — Searching for granule IDs...")
    granule_ids = search_granule_ids_for_gas_type(
        gas_type=cli_args.gas_type,
        start_date=cli_args.start_date,
        end_date=cli_args.end_date,
        max_granules=cli_args.max_granules,
        cloud_cover_min=cli_args.cloud_cover_min,
        cloud_cover_max=cli_args.cloud_cover_max,
    )

    if not granule_ids:
        print("No granules found matching the search criteria. Exiting.")
        sys.exit(0)
        
    if cli_args.reprocess_incomplete:
        remove_incomplete_granule_dirs(cli_args.output_dir, granule_ids, cli_args.gas_type)

    print(f"\nStep 2/2 — Building dataset from {len(granule_ids)} granule(s) into '{cli_args.output_dir}'...")
    run_full_dataset_build(
        granule_ids=granule_ids,
        output_dir=cli_args.output_dir,
        gas_type=cli_args.gas_type,
        cube_format=cli_args.cube_format,
        overwrite=cli_args.overwrite,
        run_mag1c=not cli_args.no_mag1c,
    )
    dataset_index_df = reconstruct_dataset_index_from_disk(cli_args.output_dir)

    print(f"\nDone. Dataset index shape: {dataset_index_df.shape}")


if __name__ == "__main__":
    argument_parser = _build_argument_parser()
    parsed_cli_args = argument_parser.parse_args()
    _run_from_cli(parsed_cli_args)
