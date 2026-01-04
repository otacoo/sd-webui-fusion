"""
SD WebUI Fusion - Checkpoint Merger Extension
Compatible with SD WebUI Classic
"""
import sys
import os

try:
    import gradio as gr
    from modules import script_callbacks, shared, sd_models
    
    # Import merger module using importlib to avoid namespace conflicts
    import importlib.util
    import os as os_module
    extension_dir = os_module.path.dirname(os_module.path.dirname(__file__))
    merger_path = os_module.path.join(extension_dir, 'scripts', 'merger.py')
    spec = importlib.util.spec_from_file_location("fusion_merger", merger_path)
    fusion_merger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fusion_merger)
    
    # Import functions from the loaded module
    get_checkpoint_files = fusion_merger.get_checkpoint_files
    merge_checkpoints = fusion_merger.merge_checkpoints
    
    # Import torch for memory management
    import torch
    
    FUSION_EXTENSION_AVAILABLE = True
except (ImportError, Exception) as e:
    FUSION_EXTENSION_AVAILABLE = False
    gr = None
    script_callbacks = None
    shared = None
    sd_models = None

# Version
FUSION_VERSION = '1.0.0'


def get_model_path(filename):
    """Get full path to checkpoint file"""
    checkpoint_dir = getattr(shared.cmd_opts, 'ckpt_dir', None) or getattr(sd_models, 'model_path', None)
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        return os.path.join(checkpoint_dir, filename)
    
    return None


def on_ui_tabs():
    """Create the UI tabs for the extension"""
    if not FUSION_EXTENSION_AVAILABLE:
        return []
    try:
        with gr.Blocks(analytics_enabled=False) as fusion_interface:
   
            gr.Markdown("## Merge Two Checkpoint Models")
                    
            with gr.Row():
                with gr.Column():
                    refresh_ckpt_btn = gr.Button("🔄 Refresh", size="sm")
                    model_a = gr.Dropdown(
                        choices=[],
                        label="Model A",
                        value=None
                    )
                    model_b = gr.Dropdown(
                        choices=[],
                        label="Model B",
                        value=None
                    )
                    
                    def refresh_checkpoints():
                        try:
                            files = get_checkpoint_files()
                            return gr.Dropdown.update(choices=files), gr.Dropdown.update(choices=files)
                        except Exception:
                            return gr.Dropdown.update(choices=[]), gr.Dropdown.update(choices=[])
                    
                    refresh_ckpt_btn.click(
                        fn=refresh_checkpoints,
                        outputs=[model_a, model_b]
                    )
                
                with gr.Column():
                    alpha = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.01,
                        label="Merge Ratio (Alpha)",
                        info="0.0 = Model A, 1.0 = Model B"
                    )
                    merge_mode = gr.Radio(
                        choices=["weight_sum", "add_difference"],
                        value="weight_sum",
                        label="Merge Mode"
                    )
                    multiplier = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Multiplier (for add_difference mode)"
                    )
            
            with gr.Row():
                output_name = gr.Textbox(
                    label="Output Filename",
                    value="merged_model.safetensors",
                    placeholder="merged_model.safetensors"
                )
                use_safetensors = gr.Checkbox(
                    label="Use SafeTensors",
                    value=True
                )
            
            unload_before_merge = gr.Checkbox(
                label="Unload models before merging",
                value=True,
                info="Unloads all models from memory before merging to avoid running out of memory."
            )
            
            merge_btn = gr.Button("Merge Checkpoints", variant="primary")
            merge_status = gr.Textbox(label="Status", interactive=False)
            
            def do_merge(model_a_name, model_b_name, alpha_val, mode, mult, out_name, safe, unload):
                if not model_a_name or not model_b_name:
                    return "Error: Please select both models"
                
                if unload:
                    try:
                        sd_models.unload_model_weights()
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                
                model_a_path = get_model_path(model_a_name)
                model_b_path = get_model_path(model_b_name)
                
                if not model_a_path or not model_b_path:
                    return "Error: Could not find model files"
                
                checkpoint_dir = shared.cmd_opts.ckpt_dir or sd_models.model_path
                output_path = os.path.join(checkpoint_dir, out_name)
                
                try:
                    result = merge_checkpoints(
                        model_a_path, model_b_path, output_path,
                        alpha=alpha_val, merge_mode=mode,
                        multiplier=mult, use_safetensors=safe
                    )
                    return f"Success! Merged model saved to: {result}"
                except Exception as e:
                    return f"Error: {str(e)}"
                    
            merge_btn.click(
                fn=do_merge,
                inputs=[model_a, model_b, alpha, merge_mode, multiplier, output_name, use_safetensors, unload_before_merge],
                outputs=merge_status
            )
            
            # Version footer
            gr.Markdown(f"<div style='text-align: center; margin-top: 20px; color: #666; font-size: 0.9em;'>v{FUSION_VERSION}</div>")
        
        return [(fusion_interface, "Fusion", "sd_webui_fusion_tab")]
    except Exception:
        return []


if FUSION_EXTENSION_AVAILABLE and script_callbacks is not None:
    try:
        if hasattr(script_callbacks, 'on_ui_tabs'):
            script_callbacks.on_ui_tabs(on_ui_tabs)
    except Exception:
        pass

