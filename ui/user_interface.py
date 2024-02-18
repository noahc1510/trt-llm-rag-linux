# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import gradio as gr
import subprocess
from ui.configuration import Configuration
import math
import uuid
import sys
import os
import kaizen_theme as kaizen
import re
import threading
import webbrowser
import socket
import random
            

class MainInterface:

    _dataset_path_key = 'dataset'
    _models_path_key = 'models'
    _dataset_directory_label_text = ".txt, .pdf, .doc files supported"
    _dataset_youtube_label_text = "YT video or playlist URLs - optionally set the video count"
    _dataset_path_updated_callback = None
    _dataset_source_updated_callback = None
    _shutdown_callback = None
    _reset_chat_callback = None
    _undo_last_chat_callback = None
    _model_change_callback = None
    _regenerate_index_callback = None
    _query_handler = None
    _state = None
    _interface = None
    _streaming = False
    _models_list = {}

    def _get_enable_disable_elemet_list(self):
        ret_val = [
            self._chat_query_input_textbox,
            self._chat_bot_window,
            self._chat_submit_button,
            self._chat_retry_button,
            self._chat_reset_button,
            self._models_dropdown,
            self._dataset_source_dropdown,
            self._dataset_update_source_edit_button,
            self._dataset_update_source_download_button,
            self._dataset_regenerate_index_button
        ]

        if self._chat_undo_button is not None:
            ret_val.append(self._chat_undo_button)
        
        return ret_val
    
    def _before_change_element_state(self, request: gr.Request):
        self._validate_session(request)
        ret_val = [
            gr.Textbox("", interactive=False),
            gr.Chatbot(),
            gr.Button(interactive=False),
            gr.Button(interactive=False),
            gr.Button(interactive=False),
            gr.Dropdown(interactive=False),
            gr.Dropdown(interactive=False),
            gr.Button(interactive=False),
            gr.Button(interactive=False),
            gr.Button(interactive=False)
        ]

        if self._chat_undo_button is not None:
            ret_val.append(gr.Button(interactive=False))

        return ret_val
    
    def _after_change_element_state(self, request: gr.Request):
        self._validate_session(request)
        ret_val = [
            gr.Textbox(interactive=True),
            [],
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.Dropdown(interactive=True),
            gr.Dropdown(interactive=True),
            gr.Button(interactive=True),
            gr.Button(interactive=True),
            gr.Button(interactive=True)
        ]

        if self._chat_undo_button is not None:
            ret_val.append(gr.Button(interactive=True))

        return ret_val
    
    def __init__(self, chatbot=None, streaming = False) -> None:
        self._interface = None
        self._query_handler = chatbot
        self._streaming = streaming
        self.config = Configuration()
        self._dataset_path = self._get_dataset_path()
        self._default_dataset_path = self._get_default_dataset_path()
        pass

    def _get_dataset_path(self):
        _dataset_path = ""
        dataset_config = self.config.get_config(self._dataset_path_key) or {}
        if 'path' in dataset_config:
            _dataset_path = dataset_config['path']
            if 'isRelative' in dataset_config and dataset_config['isRelative'] is True:
                _dataset_path = os.path.join(os.getcwd(), _dataset_path)
        
        return _dataset_path
    
    def _get_default_dataset_path(self):
        _dataset_path = ""
        dataset_config = self.config.get_config_from_file(self._dataset_path_key, "config/config.json") or {}
        if 'path' in dataset_config:
            _dataset_path = dataset_config['path']
            if 'isRelative' in dataset_config and dataset_config['isRelative'] is True:
                _dataset_path = os.path.join(os.getcwd(), _dataset_path)
        
        return _dataset_path

    def on_dataset_path_updated(self, callback):
        self._dataset_path_updated_callback = callback

    def on_dataset_source_updated(self, callback):
        self._dataset_source_updated_callback = callback

    def on_shutdown(self, callback):
        self._shutdown_callback = callback

    def on_reset_chat(self, callback):
        self._reset_chat_callback = callback

    def on_undo_last_chat(self, callback):
        self._undo_last_chat_callback = callback

    def on_model_change(self, callback):
        self._model_change_callback = callback

    def on_regenerate_index(self, callback):
        self._regenerate_index_callback = callback

    def _get_theme(self):
        primary_hue = gr.themes.Color("#76B900", "#76B900", "#76B900", "#76B900", "#76B900", "#76B900", "#76B900", "#76B900", "#76B900", "#76B900", "#76B900")
        neutral_hue = gr.themes.Color("#292929", "#292929", "#292929", "#292929", "#292929", "#292929", "#292929", "#292929", "#292929", "#292929", "#292929")
        return gr.Theme(
            primary_hue=primary_hue,
            neutral_hue=neutral_hue
        ).set(
            body_background_fill="#191919",
            body_background_fill_dark="#191919",
            block_background_fill="#292929",
            block_background_fill_dark="#292929",
            block_label_background_fill="#292929",
            block_label_background_fill_dark="#292929",
            border_color_primary="#191919",#components background
            border_color_primary_dark="#191919",
            background_fill_primary="#292929",#dropdown
            background_fill_primary_dark="#292929",
            background_fill_secondary="#393939",#response chatbot bubble
            background_fill_secondary_dark="#393939",
            color_accent_soft="#393939",#request chatbot bubble
            color_accent_soft_dark="#393939",
            #text colors
            block_label_text_color="#FFFFFF",
            block_label_text_color_dark="#FFFFFF",
            body_text_color="#FFFFFF",
            body_text_color_dark="#FFFFFF",
            body_text_color_subdued="#FFFFFF",
            body_text_color_subdued_dark="#FFFFFF",
            button_secondary_text_color="#FFFFFF",
            button_secondary_text_color_dark="#FFFFFF",
            button_primary_text_color="#FFFFFF",
            button_primary_text_color_dark="#FFFFFF",
            input_placeholder_color="#FFFFFF",#placeholder text color
            input_placeholder_color_dark="#FFFFFF",
        )

    def get_css(self):
        return kaizen.css() + open(os.path.join(os.path.dirname(__file__), 'www/app.css')).read()

    def render(self):
        with gr.Blocks(
            title="Chat with RTX",
            analytics_enabled=False,
            theme=kaizen.theme(),
            css=self.get_css(),
            js=os.path.join(os.path.dirname(__file__), 'www/app.js')
        ) as interface:
            self._interface = interface
            self._state = gr.State({})
            (
                self._shutdown_button,
                self._shutdown_post_shutdown_group,
                self._shutdown_memory_released_markdown,
                self._shutdown_invalid_session_markdown
            ) = self._render_logo_shut_down()
            with gr.Row():
                self._models_dropdown, self._models_group = self._render_models()
                (
                    self._dataset_source_textbox,
                    self._dataset_update_source_edit_button,
                    self._dataset_update_source_download_button,
                    self._dataset_source_dropdown,
                    self._dataset_regenerate_index_button,
                    self._dataset_view_transcripts_button,
                    self._dataset_label_markdown,
                    self._dataset_group,
                    self._dataset_youtube_video_count
                ) = self._render_dataset_picker()
            (
                self._sample_question_components,
                self._sample_question_rows,
                self._sample_question_empty_space_component,
                self._sample_qustion_default_dataset_markdown
            ) = self._render_sample_question()
            (
                self._chat_bot_window,
                self._chat_query_input_textbox,
                self._chat_submit_button,
                self._chat_retry_button,
                self._chat_undo_button,
                self._chat_reset_button,
                self._chat_query_group,
                self._chat_disclaimer_markdown
            ) = self._render_chatbot(show_chatbot=len(self._sample_question_components) == 0)
            self._handle_events()
            self._handle_links()
        interface.queue()
        port = self._get_free_port()
        self._open_app(port)
        interface.launch(
            favicon_path=os.path.join(os.path.dirname(__file__), 'assets/nvidia_logo.png'),
            show_api=False,
            server_port=port
        )

    def _get_free_port(self):
        # Create a socket object
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Set a short timeout for the connection attempt
        sock.settimeout(1)
        port = None
        while port is None:
            port = random.randint(1024, 49000)
            try:
                # Attempt to bind to the port
                sock.bind(("127.0.0.1", port))
            except OSError as e:
                port = None
                if e.errno != 98:  # errno 98: Address already in use
                    print('OS error', e)
                    break 
        sock.close()
        return port

    def _open_app(self, port):
        def launch_thread(cookie):
            launch_url = f'http://127.0.0.1:{port}?cookie={cookie}&__theme=dark'
            print(f'Open {launch_url} in browser to start Chat with RTX')
            webbrowser.open(launch_url)
            return None
        
        self._secure_cookie = str(uuid.uuid4())
        threading.Thread(target=launch_thread, args=(self._secure_cookie,)).start()
        return None

    def _validate_request(self, request: gr.Request):
        headers = request.headers
        session_key = None
        if 'cookie' in headers:
            cookies = headers['cookie']
            if '_s_chat_=' in cookies:
                cookies = cookies.split('; ')
                for i, cookie in enumerate(cookies):
                    key, value = cookie.split('=')
                    if key == '_s_chat_':
                        session_key = value
        
        if session_key == None or session_key != self._secure_cookie:
            raise 'session validation failed'
        
        return True
                    
    
    def _handle_links(self):
        def open_file(file, request: gr.Request):
            print('handle link call', file)
            self._validate_request(request)
            try:
                pattern = re.compile(r'^file:/+')
                file = re.sub(pattern, '', file)
                file = os.path.normpath(file)
                from urllib.parse import unquote
                file = unquote(file)
                command = f'start "" "{file}"'
                subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            except Exception as e:
                print(f"Error: {e}")

        file_path = gr.Textbox(visible=False)
        gr.Button(visible=False).click(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            open_file,
            file_path,
            None,
            api_name="open_file"
        )

    def _get_session_id(self, state):
        if isinstance(state, object):
            if not 'session-id' in state:
                state['session-id'] = str(uuid.uuid4())
            return state['session-id']
        return None

    def _render_models(self):
        models = self.config.get_config(self._models_path_key)
        model_names = []
        for model in models['supported']:
            if model['installed'] is True:
                model_names.append(model['name'])
                self._models_list[model['name']] = model

        with gr.Column():
            with gr.Group(elem_classes="padding-8p model-goup") as models_group:
                gr.Markdown("<b>AI model</b>")
                gr.Markdown(
                    'Select AI model',
                    elem_classes="description-secondary-markdown"
                )
                def get_selected_value():
                    selected = self.config.get_config('models/selected')
                    if len(self._models_list) > 0:
                        if not selected in self._models_list:
                            selected = self.config.get_config_from_file('models/selected', "config/config.json")
                        if not selected in self._models_list:
                            selected = list(self._models_list.keys())[0]
                    return selected
                return gr.Dropdown(
                    model_names,
                    elem_classes="height-40p",
                    value=get_selected_value,
                    container=False,
                    filterable=False
                ), models_group

    def _render_logo_shut_down(self):
        with gr.Row():
            gr.Image(os.path.join(os.path.dirname(__file__), "assets/nvidia_logo.png"),
                interactive=False,
                show_label=False,
                show_download_button=False,
                width=40,
                scale=0,
                container=False,
                min_width=40
            )
            gr.HTML("""
                <h1 style="font-size:32px; line-height:40px; margin:0; padding:0">Chat with RTX</h1>
            """)
            shutdown_button = gr.Button(
                "",
                scale=0,
                icon=os.path.join(os.path.dirname(__file__), 'assets/shutdown.png'),
                elem_classes="icon-button tooltip-component",
                elem_id="shutdown-btn"
            )
        
        with gr.Group(visible=False, elem_classes="shutdown-group") as post_shutdown_group:
            with gr.Row():
                gr.HTML("")
                gr.Image(os.path.join(os.path.dirname(__file__), "assets/info.png"),
                    interactive=False,
                    show_label=False,
                    show_download_button=False,
                    width=40,
                    scale=0,
                    container=False,
                    min_width=40
                )
                gr.HTML("")
            with gr.Row():
                shutdown_memory_released_markdown = gr.Markdown(
                    "Video memory released. Reopen RTX Chat from desktop to continue chatting.",
                    elem_classes="text-align-center"
                )
                shutdown_invalid_session_markdown = gr.Markdown(
                    "Invalid session. Reopen RTX Chat from desktop to continue chatting.",
                    elem_classes="text-align-center"
                )

        return shutdown_button, post_shutdown_group, shutdown_memory_released_markdown, shutdown_invalid_session_markdown

    def _render_dataset_picker(self):
        sources = self.config.get_config("dataset/sources")
        self._dataset_selected_source = self.config.get_config("dataset/selected")
        with gr.Column(elem_classes="dataset-column"):
            with gr.Group(elem_classes="padding-8p dataset-goup") as dataset_group:
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("<b>Dataset</b>")
                        dataset_label_markdown = gr.Markdown(
                            self._dataset_directory_label_text if self._dataset_selected_source=="directory" else self._dataset_youtube_label_text,
                            elem_classes="description-secondary-markdown",
                            elem_id="dataset-description-label"
                        )
                    view_transcripts_button = gr.Button(
                        "", 
                        icon=os.path.join(os.path.dirname(__file__), 'assets/folder.png'),
                        elem_classes="icon-button tooltip-component margin-right-8p",
                        elem_id="dataset-view-transcript-folder-button",
                        scale=0,
                        visible=self._dataset_selected_source=="youtube"
                    )
                    regenerate_vector_button = gr.Button(
                        "", 
                        icon=os.path.join(os.path.dirname(__file__), 'assets/regenerate.png'),
                        elem_classes="icon-button tooltip-component",
                        elem_id="dataset-regenerate-index-btn",
                        scale=0
                    )
                dataset_source_dropdown = gr.Dropdown(
                    self.config.get_display_strings(sources),
                    value=lambda: self.config.get_display_strings(self._dataset_selected_source),
                    show_label=False,
                    container=False,
                    filterable=False,
                    elem_classes="margin-bottom-8p height-40p"
                )
                with gr.Row():
                    dataset_source_textbox = gr.Textbox(
                        lambda: self._dataset_path,
                        scale=9,
                        container=False,
                        elem_classes="height-40p margin-right-8p",
                        interactive=self._dataset_selected_source=="youtube",
                        placeholder="Enter URL...",
                        max_lines=1,
                        autoscroll=True
                    )
                    dataset_youtube_video_count_textbox = gr.Number(
                        25,
                        scale=1,
                        container=False,
                        elem_classes="height-40p margin-right-8p",
                        visible=self._dataset_selected_source=="youtube"
                    )
                    dataset_update_source_edit_button = gr.Button(
                        "",
                        icon=os.path.join(os.path.dirname(__file__), 'assets/edit.png'),
                        elem_classes="icon-button tooltip-component",
                        elem_id="dataset-update-source-edit-button",
                        visible=self._dataset_selected_source=="directory",
                        scale=0
                    )
                    dataset_update_source_download_button = gr.Button(
                        "",
                        icon=os.path.join(os.path.dirname(__file__), 'assets/download.png'),
                        elem_classes="icon-button tooltip-component",
                        elem_id="dataset-update-source-download-button",
                        visible=self._dataset_selected_source=="youtube",
                        scale=0
                    )
                    return (
                        dataset_source_textbox,
                        dataset_update_source_edit_button,
                        dataset_update_source_download_button,
                        dataset_source_dropdown,
                        regenerate_vector_button,
                        view_transcripts_button,
                        dataset_label_markdown,
                        dataset_group,
                        dataset_youtube_video_count_textbox
                    )


    def _render_sample_question(self):
        question_butons = []
        question_rows = []
        sample_questions: list = self.config.get_config("sample_questions")
        if sample_questions is None or len(sample_questions) == 0:
            return question_butons, question_rows
        
        chat_window_height = 500
        row_height = 16
        button_height = 42
        elem_per_row = 2
        row_count = math.ceil(len(sample_questions) / elem_per_row)
        height = chat_window_height - (row_count*(row_height + button_height))
        empty_space_component = gr.HTML(f'<div style="height: {height}px"></div>')
        default_dataset_label = gr.Markdown(
            "Default dataset is a sampling of articles recently published on GeForce News",
            elem_classes="description-secondary-markdown chat-disclaimer-message margin-"
        )
        for i in range(0, len(sample_questions), 2):
            row_questions = sample_questions[:2]
            sample_questions = sample_questions[2:]
            with gr.Row() as question_row:
                for index, question in enumerate(row_questions):
                    query = question["query"]
                    button = gr.Button(
                        query,
                        elem_classes="sample-question-button"
                    )
                    question_butons.append({
                        "question": query,
                        "component": button
                    })
                    question_rows.append(question_row)
        return question_butons, question_rows, empty_space_component, default_dataset_label

    def _render_chatbot(self, show_chatbot):
        chatbot_window = gr.Chatbot(
            show_label=False,
            elem_classes="chat-window",
            visible=show_chatbot,
            elem_id="main-chatbot-window",
            sanitize_html=False
        )
        with gr.Group() as query_group:
            with gr.Row():
                query_input = gr.Textbox(placeholder="Chat with RTX...", scale=9, container=False)
                submit_button = gr.Button("SEND", variant="primary", scale=1)
        with gr.Row():
            gr.HTML("")
            retry_button = gr.Button(
                "",
                elem_classes="icon-button tooltip-component",
                elem_id="chatbot-retry-button",
                scale=0,
                icon=os.path.join(os.path.dirname(__file__), 'assets/retry.png'),
            )
            undo_button = None
            if self.config.get_config_from_file("is_chat_engine", os.path.join(os.path.curdir, "config/app_config.json")) == False:
                undo_button = gr.Button(
                    "",
                    scale=0,
                    icon=os.path.join(os.path.dirname(__file__), 'assets/undo.png'),
                    elem_classes="icon-button tooltip-component",
                    elem_id="chatbot-undo-button"
                )
            reset_button = gr.Button(
                "",
                elem_classes="icon-button tooltip-component",
                elem_id="chatbot-reset-button",
                scale=0,
                icon=os.path.join(os.path.dirname(__file__), 'assets/reset.png'),
            )
            gr.HTML("")
        chat_disclaimer_markdown = gr.Markdown(
            "Chat with RTX response quality depends on the AI model's accuracy and the input dataset. Please verify important information.",
            elem_classes="description-secondary-markdown chat-disclaimer-message margin-"
        )
        return (chatbot_window, query_input, submit_button, retry_button, undo_button, reset_button, query_group, chat_disclaimer_markdown)

    def _handle_events(self):
        self._handle_load_events()
        self._handle_shutdown_events()
        self._handle_model_events()
        self._handle_dataset_events()
        self._handle_chatbot_events()
        return None

    def _validate_session_and_raise(self, request: gr.Request):
        try:
            self._validate_request(request)
        except Exception as e:
            raise gr.Error('Invalid session')

    def _validate_session(self, request: gr.Request):
        try:
            self._validate_request(request)
        except Exception as e:
            return [
                gr.Group(visible=False),
                gr.Group(visible=False),
                gr.Chatbot(visible=False),
                gr.Group(visible=False),
                gr.Button(visible=False),
                gr.Button(visible=False),
                gr.Group(visible=True),
                gr.Button(visible=False),
                gr.Button(visible=False),
                gr.Markdown(visible=False),
                gr.Markdown(visible=True),
                gr.Markdown(visible=False)
             ] + self._get_sample_question_components_new(True)
        return [
            gr.Group(),
            gr.Group(),
            gr.Chatbot(),
            gr.Group(),
            gr.Button(),
            gr.Button(),
            gr.Group(),
            gr.Button(),
            gr.Button(),
            gr.Markdown(),
            gr.Markdown(),
            gr.Markdown()
         ] + self._get_sample_question_components_new()
    
    def _get_validate_session_output(self):
        return [
            self._models_group,
            self._dataset_group,
            self._chat_bot_window,
            self._chat_query_group,
            self._chat_reset_button,
            self._chat_retry_button,
            self._shutdown_post_shutdown_group,
            self._shutdown_button,
            self._chat_undo_button,
            self._chat_disclaimer_markdown,
            self._shutdown_invalid_session_markdown,
            self._shutdown_memory_released_markdown
        ] + self._get_sample_question_components()

    def _handle_load_events(self):
        self._interface.load(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            self._show_hide_sample_questions,
            self._get_show_hide_sample_questions_inputs(),
            self._get_show_hide_sample_questions_outputs()
        )
        return None

    def _handle_shutdown_events(self):
        def close_thread(session_id):
            if self._shutdown_callback:
                self._shutdown_callback(session_id)
            self._interface.close()
            self._interface = None
            print('exiting')
            os._exit(0)
            
        def handle_shutdown(state, request: gr.Request):
            self._validate_session(request)
            if self._interface is not None:
                _close_thread = threading.Thread(target=close_thread, args=(self._get_session_id(state),))
                _close_thread.start()
            else:
                print("Interface not initialized or already closed")
            return state
        def before_shutdown(request: gr.Request):
            self._validate_session(request)
            ret_val = [
                gr.Group(visible=False),
                gr.Group(visible=False),
                gr.Chatbot(visible=False),
                gr.Group(visible=False),
                gr.Button(visible=False),
                gr.Button(visible=False),
                gr.Group(visible=True),
                gr.Button(visible=False),
                gr.Button(visible=False),
                gr.Markdown(visible=False),
                gr.Markdown(visible=False),
                gr.Markdown(visible=True)
            ] + self._get_sample_question_components_new(True)
            return ret_val
        

        self._shutdown_button.click(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            before_shutdown,
            None,
            [
                self._models_group,
                self._dataset_group,
                self._chat_bot_window,
                self._chat_query_group,
                self._chat_reset_button,
                self._chat_retry_button,
                self._shutdown_post_shutdown_group,
                self._shutdown_button,
                self._chat_undo_button,
                self._chat_disclaimer_markdown,
                self._shutdown_invalid_session_markdown,
                self._shutdown_memory_released_markdown
            ] + self._get_sample_question_components()
        ).then(
            handle_shutdown,
            self._state,
            self._state
        )

    def _handle_model_events(self):
        
        def on_selection_change(newModel, state, request: gr.Request):
            self._validate_session(request)
            if self._model_change_callback:
                self._model_change_callback(
                    self._models_list[newModel]['name'],
                    self._models_list[newModel]['metadata'],
                    self._get_session_id(state)
                )
            self.config.set_config("models/selected", newModel)
            return newModel, state
        
        self._models_dropdown.change(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            self._before_change_element_state,
            None,
            self._get_enable_disable_elemet_list()
        ).then(
            on_selection_change,
            [self._models_dropdown, self._state],
            [self._models_dropdown, self._state]
        ).then(
            self._after_change_element_state,
            None,
            self._get_enable_disable_elemet_list(),
            show_progress=False
        )

    def _can_show_youtube_video_count_warning(self, count, state):
        if isinstance(state, object):
            if not 'video-count-warning-shown' in state and count > 25:
                state['video-count-warning-shown'] = True
                return True
        return False
    
    def _handle_dataset_events(self):
        
        def select_folder(path, video_count, state, request: gr.Request):
            self._validate_session(request)
            if self._dataset_selected_source == "directory":
                command = [sys.executable, "./ui/select_folder.py"]
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output, _ = process.communicate()
                # Check if the command was successful
                result_string = ""
                if process.returncode == 0:
                    result_string = output.decode().strip()
                else:
                    print("Error executing script:", process.returncode)
                if len(result_string) > 0:
                    self._dataset_path = result_string
                    self.config.set_config(self._dataset_path_key, {"path": self._dataset_path, "isRelative": False})
            else:
                self._dataset_path = path

            if self._dataset_path_updated_callback:
                self._dataset_path_updated_callback(
                    self._dataset_selected_source,
                    self._dataset_path,
                    video_count if self._dataset_selected_source == "youtube" else None,
                    self._get_session_id(state)
                )
            return self._dataset_path, state
        
        gr.on(
            [self._dataset_update_source_edit_button.click, self._dataset_update_source_download_button.click],
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            self._before_change_element_state,
            None,
            self._get_enable_disable_elemet_list()
        ).then(
            select_folder,
            [self._dataset_source_textbox, self._dataset_youtube_video_count, self._state],
            [self._dataset_source_textbox, self._state]
        ).then(
            self._after_change_element_state,
            None,
            self._get_enable_disable_elemet_list(),
            show_progress=False
        ).then(
            self._show_hide_sample_questions,
            self._get_show_hide_sample_questions_inputs(),
            self._get_show_hide_sample_questions_outputs(),
            show_progress=False
        )

        def on_dataset_source_changed(source, state, request: gr.Request):
            self._validate_session(request)
            self._dataset_selected_source = self.config.get_display_string_keys(source)
            source = self._dataset_selected_source
            self._dataset_path = self._get_dataset_path() if source=="directory" else ""
            if self._dataset_source_updated_callback:
                self._dataset_source_updated_callback(
                    self._dataset_selected_source,
                    "" if source=="youtube" else self._dataset_path,
                    self._get_session_id(state)
                )
            return [
                gr.Textbox(
                    interactive=source=="youtube",
                    visible=source!="nodataset",
                    value="" if source=="youtube" else self._dataset_path
                ),
                gr.Button(visible=source=="directory"),
                gr.Button(visible=source=="youtube"),
                gr.Number(visible=source=="youtube"),
                gr.Button(visible=source!="nodataset"),
                gr.Button(visible=source=="youtube"),
                state
            ]
        
        change_source_description = """
            (source) => {
                let element = document.querySelector('[id="dataset-description-label"]');
                let label = source === "Folder Path" ? ".txt, .pdf, .doc files supported" : "YT video or playlist URLs - optionally set the video count";
                if(element) {
                    element.querySelector('p').textContent = label;
                } else {
                    console.error("failed to update");
                }
                return source;
            }
        """

        self._dataset_source_dropdown.change(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            self._before_change_element_state,
            None,
            self._get_enable_disable_elemet_list(),
            show_progress=False
        ).then(            
            on_dataset_source_changed,
            [self._dataset_source_dropdown, self._state],
            [
                self._dataset_source_textbox,
                self._dataset_update_source_edit_button,
                self._dataset_update_source_download_button,
                self._dataset_youtube_video_count,
                self._dataset_regenerate_index_button,
                self._dataset_view_transcripts_button,
                self._state
            ],
            show_progress=False
        ).then(
            lambda x: x,
            self._dataset_source_dropdown,
            self._dataset_source_dropdown,
            show_progress=False,
            js=change_source_description
        ).then(
            self._after_change_element_state,
            None,
            self._get_enable_disable_elemet_list(),
            show_progress=False
        ).then(
            self._show_hide_sample_questions,
            self._get_show_hide_sample_questions_inputs(),
            self._get_show_hide_sample_questions_outputs(),
            show_progress=False
        )

        def regenerate_index(state, request: gr.Request):
            self._validate_session(request)
            if self._regenerate_index_callback:
                self._regenerate_index_callback(self._dataset_selected_source, self._dataset_path, self._get_session_id(state))
            return self._dataset_path, state

        self._dataset_regenerate_index_button.click(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            self._before_change_element_state,
            None,
            self._get_enable_disable_elemet_list()
        ).then(
            regenerate_index,
            self._state,
            [self._dataset_source_textbox, self._state]
        ).then(
            self._after_change_element_state,
            None,
            self._get_enable_disable_elemet_list(),
            show_progress=False
        )

        def show_video_count_warning(count, state, request: gr.Request):
            self._validate_session(request)
            if self._can_show_youtube_video_count_warning(count, state):
                gr.Warning("Including over 25 videos may extend processing time and reduce response accuracy.")
            return state

        self._dataset_youtube_video_count.change(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            show_video_count_warning,
            [self._dataset_youtube_video_count, self._state],
            self._state
        )

        def open_transcripts_folder(request: gr.Request):
            self._validate_session(request)
            subprocess.Popen([sys.executable, "./ui/open_folder.py", os.path.join(os.getcwd(), "youtube_dataset")])
            return None

        self._dataset_view_transcripts_button.click(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            open_transcripts_folder,
            None,
            None
        )


# dataset events ends
    def _show_hide_sample_questions(self, query, history, dataset_source, state, request: gr.Request):
        self._validate_session(request)
        dataset_source = self.config.get_display_string_keys(dataset_source)
        sample_question_shown = state['sample_question_shown'] if isinstance(state, object) and 'sample_question_shown' in state else False
        hide_sample_ques = \
            len(query) > 0 or \
            len(history) > 0 or \
            dataset_source == "youtube" or \
            sample_question_shown or \
            (os.path.normpath(self._dataset_path) != os.path.normpath(self._default_dataset_path))
        if isinstance(state, object):
            state['sample_question_shown'] = hide_sample_ques
        ret_val = [gr.Button(visible=not hide_sample_ques) for _ in self._sample_question_components]
        ret_val.insert(0, gr.Chatbot(history, visible=hide_sample_ques))
        [ret_val.append(gr.Row(visible=not hide_sample_ques)) for _ in self._sample_question_rows]
        ret_val.append(gr.HTML(visible=not hide_sample_ques))
        ret_val.append(gr.Markdown(visible=not hide_sample_ques))
        ret_val.append(state)
        return ret_val
    
    def _get_show_hide_sample_questions_inputs(self):
        return [
            self._chat_query_input_textbox ,self._chat_bot_window, self._dataset_source_dropdown, self._state
        ]
    
    def _get_sample_question_components_new(self, hide_sample_ques: bool = None):
        if hide_sample_ques is None: # neither show nor hide
            ret_val = [gr.Button() for _ in self._sample_question_components]
            [ret_val.append(gr.Row()) for _ in self._sample_question_rows]
            ret_val.append(gr.HTML())
            ret_val.append(gr.Markdown())
        else:
            ret_val = [gr.Button(visible=not hide_sample_ques) for _ in self._sample_question_components]
            [ret_val.append(gr.Row(visible=not hide_sample_ques)) for _ in self._sample_question_rows]
            ret_val.append(gr.HTML(visible=not hide_sample_ques))
            ret_val.append(gr.Markdown(visible=not hide_sample_ques))
        return ret_val
            

    def _get_sample_question_components(self):
        sample_questions_buttons = [question['component'] for question in self._sample_question_components]
        return sample_questions_buttons + self._sample_question_rows + [self._sample_question_empty_space_component, self._sample_qustion_default_dataset_markdown]

    def _get_show_hide_sample_questions_outputs(self):
        return [self._chat_bot_window] + self._get_sample_question_components() + [self._state]

# chat bot events
    def _handle_chatbot_events(self):
        def process_input(query, history, request: gr.Request):
            self._validate_session(request)
            if len(query) == 0:
                return "", history
            history.append([query, None])
            return "", history
        
        def process_output(history, state, request: gr.Request):
            self._validate_session(request)
            if len(history) == 0:
                yield history, state
            else:
                query = history[-1]
                if query[1] != None:
                    yield history, state
                elif self._query_handler:
                    for response in self._query_handler(query[0], history[:-1], self._get_session_id(state)):
                        history[-1][1] = response
                        yield history, state
                else:
                    history[-1][1] = "ChatBot not ready..."
                    yield history, state
            
        #undo handler
        def process_undo_last_chat(history: list, state, request: gr.Request):
            self._validate_session(request)
            if len(history) == 0:
                return history, state

            history = history[:len(history) - 1]
            if self._undo_last_chat_callback:
                self._undo_last_chat_callback(history, self._get_session_id(state))
            
            return history, state

        #retry handler
        def process_retry(history: list, request: gr.Request):
            self._validate_session(request)
            if len(history) == 0:
                return history

            lastChat = history[-1]
            history = history[:len(history) - 1]
            _, history = process_input(lastChat[0], history, request)
            return history

        def reset(state, request: gr.Request):
            self._validate_session(request)
            if self._reset_chat_callback:
                self._reset_chat_callback(self._get_session_id(state))
            return "", [], state

        gr.on(
            [self._chat_query_input_textbox.submit, self._chat_submit_button.click],
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            self._show_hide_sample_questions,
            self._get_show_hide_sample_questions_inputs(),
            self._get_show_hide_sample_questions_outputs()
        ).then(
            process_input,
            [self._chat_query_input_textbox, self._chat_bot_window], 
            [self._chat_query_input_textbox, self._chat_bot_window]
        ).then(
            process_output,
            [self._chat_bot_window, self._state],
            [self._chat_bot_window, self._state]
        )

        self._chat_retry_button.click(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            process_retry,
            [self._chat_bot_window],
            [self._chat_bot_window]
        ).then(
            process_output,
            [self._chat_bot_window, self._state],
            [self._chat_bot_window, self._state]
        )

        if self._chat_undo_button:
            self._chat_undo_button.click(
                self._validate_session,
                None,
                self._get_validate_session_output()
            ).then(
                self._validate_session_and_raise,
                None,
                None
            ).success(
                process_undo_last_chat,
                [self._chat_bot_window, self._state],
                [self._chat_bot_window, self._state]
            )
        self._chat_reset_button.click(
            self._validate_session,
            None,
            self._get_validate_session_output()
        ).then(
            self._validate_session_and_raise,
            None,
            None
        ).success(
            reset, 
            self._state,
            [self._chat_query_input_textbox, self._chat_bot_window, self._state]
        )

        def handle_sample_question_click(evt: gr.EventData, request: gr.Request):
            self._validate_session(request)
            return evt.target.value
        
        for sample in self._sample_question_components:
            button: gr.Button = sample['component']
            button.click(
                self._validate_session,
                None,
                self._get_validate_session_output()
            ).then(
                self._validate_session_and_raise,
                None,
                None
            ).success(
                handle_sample_question_click,
                None,
                self._chat_query_input_textbox
            ).then(
                self._show_hide_sample_questions,
                self._get_show_hide_sample_questions_inputs(),
                self._get_show_hide_sample_questions_outputs()
            ).then(
                process_input,
                [self._chat_query_input_textbox, self._chat_bot_window], 
                [self._chat_query_input_textbox, self._chat_bot_window]
            ).then(
                process_output,
                [self._chat_bot_window, self._state],
                [self._chat_bot_window, self._state]
            )
