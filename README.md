# SD WebUI Fusion - Checkpoint Merger Extension

A very simple SD WebUI extension for merging two checkpoint models. Compatible with SD WebUI Classic.

## Features

- Merge two checkpoint models with various merge modes
- Weight Sum: Simple weighted combination
- Add Difference: Add scaled difference between models

## SD WebUI Installation

1. Go into `Extensions` tab > `Install from URL`
2. Paste `https://github.com/otacoo/sd-webui-civitai-downloader.git`
3. Press Install
4. Apply and Restart the UI

## Usage

1. Go to the "Fusion" tab
2. Select two checkpoint models (Model A and Model B)
3. Adjust the merge ratio (alpha) - 0.0 = Model A, 1.0 = Model B
4. Choose merge mode
5. Enter output filename
6. Optionally enable "Unload models before merging" to free memory
7. Click "Merge Checkpoints"
