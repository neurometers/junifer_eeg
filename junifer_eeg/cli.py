"""Command line interface for junifer_eeg."""

import click
from junifer.api import run as junifer_run


@click.group()
@click.version_option()
def cli():
    """JUnifer EEG - EEG extension for junifer."""
    pass


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def run(config_file, verbose):
    """Run EEG feature extraction using a configuration file.

    CONFIG_FILE: Path to the YAML configuration file.
    """
    if verbose:
        click.echo(f"Running junifer_eeg with config: {config_file}")

    try:
        junifer_run(config_file)
        click.echo("✓ Analysis completed successfully!")
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort() from e


@cli.command()
def info():
    """Show information about available components."""
    click.echo("Available EEG Data Grabbers:")
    click.echo("  - EEGDataGrabber: Grab EEG files (EDF format)")
    click.echo()
    click.echo("Available EEG Preprocessors:")
    click.echo("  - EEGLoader: Load EEG files into MNE Raw objects")
    click.echo()
    click.echo("Available EEG Markers:")
    click.echo("  - SpectralPower: Compute power in frequency bands")


if __name__ == "__main__":
    cli()
