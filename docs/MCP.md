# MCP server

English · [Italiano](it/MCP.md)

Excido speaks the [Model Context Protocol](https://modelcontextprotocol.io), so Claude (Claude Code, Claude Desktop) and other MCP clients can remove backgrounds for you. You can ask things like *"cut out every photo in the shoes folder and put them on white"* or *"the hair got lost, try a matting model"*. Everything still runs on your computer.

## Two ways to connect

| Mode | Command | Best for |
|---|---|---|
| **HTTP**, built into the web app | `uv run excido`, then connect to `http://127.0.0.1:7860/mcp` | Claude Code and any client that supports Streamable HTTP. It shares the app's model and history, so what the agent does appears in the web UI. |
| **stdio** | `uv run excido mcp` | Claude Desktop and clients that launch a local command. If the web app is running it forwards the work to it (one process owns the GPU); otherwise it loads the model itself. |

### Claude Code

```bash
claude mcp add --transport http excido http://127.0.0.1:7860/mcp
```

Or, without keeping the web app open:

```bash
claude mcp add excido -- uv run --directory /path/to/excido excido mcp
```

### Claude Desktop

Edit `claude_desktop_config.json` (*Settings → Developer → Edit config*):

```json
{
  "mcpServers": {
    "excido": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/excido", "excido", "mcp"]
    }
  }
}
```

On Windows, write the path as `C:\\Users\\you\\excido`. If Claude Desktop can't find `uv`, put the full path to the `uv` executable in `command`.

## The workspace folder

The tools can **only read images from one folder**, and **only write into its `excido-output/` subfolder**.

- **Default**: `~/Pictures/Excido` (`C:\Users\<you>\Pictures\Excido` on Windows). It is created on first use.
- **Change it**: use `--mcp-root <folder>` with `excido serve` or `excido mcp`, or set the `EXCIDO_MCP_ROOT` environment variable.
- **How it is enforced**:
  - Paths are relative to the workspace.
  - `..`, absolute paths elsewhere, and symbolic links that point outside are refused.
  - Results never overwrite existing files: a `-1`, `-2`… suffix is added instead.
- **The web app is not affected**: it keeps accepting whatever you upload by hand.

Put your photos in the workspace, then ask the agent to work on them. Results appear in `excido-output/`, as `<name>_cutout.png` or `<name>_composite.png`, plus `<name>_mask.png`.

## Tools

| Tool | What it does |
|---|---|
| `workspace_info` | Shows the workspace path, the output folder, the image count and which options are enabled. |
| `list_images` | Lists the images in the workspace or one of its subfolders. |
| `remove_background` | Processes one image, with `model`, `resolution`, `max_side`, `precision`, `device`, `mask_mode`, `threshold`, `invert`, `grow` (±px, 0.5 steps), `refine`, `background`, `output_format` and `preview`. It returns the output paths, a `run_id`, timings and `foreground_ratio`. |
| `remove_background_batch` | Same as `remove_background`, for a list of images or a whole folder. |
| `recompose` | Changes mask or background settings of a previous result, including `grow`, without re-running the model (fast). |
| `list_models` | Lists the 16 variants: what each is for, size, and whether it is downloaded. |
| `download_model` | Downloads a model from the pinned official catalog. It is **disabled by default**. |
| `system_info` | Shows the devices (CUDA, Metal, CPU), memory and the loaded model. |

`background` accepts four values:
- `transparent`;
- a color such as `#ffffff`;
- `blur` or `blur:30`;
- the path of another workspace image to place behind the subject.

`psd=true` also writes `<name>_layers.psd` into `excido-output/`: the subject with an editable layer mask, the hidden original and the background. It works with `remove_background`, `remove_background_batch` and `recompose`.

`preview=true` also returns a small JPEG of the result, with transparency shown on a checkerboard, so the agent can look at it and decide whether to retry, for example with a matting model. Previews cost tokens, so they are off by default.

## Options

| Flag (`serve` / `mcp`) | Environment variable | Default |
|---|---|---|
| `--mcp-root <folder>` | `EXCIDO_MCP_ROOT` | `~/Pictures/Excido` |
| `--mcp-allow-urls` | `EXCIDO_MCP_ALLOW_URLS=1` | off: only workspace files |
| `--mcp-allow-downloads` | `EXCIDO_MCP_ALLOW_DOWNLOADS=1` | off: the user runs `excido download` |
| `--no-mcp` (`serve` only) | `EXCIDO_MCP=0` | the `/mcp` endpoint is on |
| `--standalone` (`mcp` only) | `EXCIDO_MCP_STANDALONE=1` | reuse the running web app |

## Security notes

- **Everything stays on the machine that runs Excido.** Publishing or cloning the repository exposes nothing: each user runs their own copy, on their own computer.
- **The `/mcp` endpoint answers only `localhost` requests.** It checks the `Host` and `Origin` headers, which blocks DNS-rebinding attacks from web pages. This holds even if the web app is started with `--host 0.0.0.0`.
- **The whole web app rejects foreign `Host` headers** for the same reason.
- **MCP clients usually ask you before each tool call.** Keep that confirmation on for tools that write files.
