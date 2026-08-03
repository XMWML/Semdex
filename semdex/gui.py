"""Cross-platform native desktop interface for Semdex.

The module imports Qt only when the desktop command is launched.  Semdex's
command line, API, and test suite therefore remain usable on minimal/headless
Python installations.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Callable

from .desktop import DesktopController


class SearchRequestGuard:
    """Reject a delayed search response after the user has issued a newer one."""

    def __init__(self) -> None:
        self._generation = 0

    def begin(self) -> int:
        self._generation += 1
        return self._generation

    def is_current(self, generation: int) -> bool:
        return generation == self._generation


def run_gui(config_path: str | None = None) -> int:
    """Start the native GUI and return its process exit code."""
    try:
        from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
        from PySide6.QtGui import QAction, QDesktopServices
        from PySide6.QtWidgets import (
            QApplication,
            QButtonGroup,
            QCheckBox,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QFrame,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QHeaderView,
            QLabel,
            QLineEdit,
            QListWidget,
            QMainWindow,
            QMessageBox,
            QPlainTextEdit,
            QPushButton,
            QScrollArea,
            QSpinBox,
            QTabWidget,
            QTreeWidget,
            QTreeWidgetItem,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:
        print(
            "桌面界面需要 Qt 运行时。请执行 `uv sync --extra gui`，"
            "或在 macOS 双击 Start Semdex.command。",
            file=sys.stderr,
        )
        return 1

    class TaskBridge(QObject):
        finished = Signal(object, object)

    class SettingsDialog(QDialog):
        """Native form matching the values exposed by the WebUI settings page."""

        MODEL_LABELS = [
            ("llm", "默认 LLM"),
            ("agent", "检索 Agent"),
            ("entities", "实体抽取"),
            ("fallback", "提取兜底"),
            ("vision", "图片理解"),
            ("embedding", "语义嵌入"),
        ]
        MODEL_CAPABILITIES = {
            "llm": "chat",
            "agent": "chat",
            "entities": "chat",
            "fallback": "chat",
            "vision": "vision",
            "embedding": "embedding",
        }

        def __init__(self, controller: DesktopController, parent: QWidget | None = None):
            super().__init__(parent)
            self.controller = controller
            self.start_after_save = False
            self._model_catalog: dict[str, Any] = {}
            self._model_bridges: list[TaskBridge] = []
            self.setWindowTitle("Semdex 设置")
            self.resize(880, 720)

            outer = QVBoxLayout(self)
            tabs = QTabWidget()
            tabs.addTab(self._build_storage_tab(), "索引范围")
            tabs.addTab(self._build_models_tab(), "模型")
            tabs.addTab(self._build_tools_tab(), "OCR 与语音")
            tabs.addTab(self._build_features_tab(), "功能")
            outer.addWidget(tabs)

            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
            self.save_button = QPushButton("保存设置")
            self.save_index_button = QPushButton("保存并开始索引")
            buttons.addButton(self.save_button, QDialogButtonBox.ButtonRole.ActionRole)
            buttons.addButton(self.save_index_button, QDialogButtonBox.ButtonRole.AcceptRole)
            buttons.rejected.connect(self.reject)
            self.save_button.clicked.connect(self.save)
            self.save_index_button.clicked.connect(self.save_and_index)
            outer.addWidget(buttons)
            self.load_settings(controller.settings())
            self.refresh_local_models()

        @staticmethod
        def _form_widget(layout: QFormLayout, label: str, widget: QWidget) -> QWidget:
            layout.addRow(label, widget)
            return widget

        @staticmethod
        def _int_box(minimum: int, maximum: int = 2_147_483_647) -> QSpinBox:
            box = QSpinBox()
            box.setRange(minimum, maximum)
            return box

        @staticmethod
        def _set_combo(combo: QComboBox, value: str) -> None:
            index = combo.findData(value)
            if index < 0:
                combo.addItem(value, value)
                index = combo.count() - 1
            combo.setCurrentIndex(index)

        @staticmethod
        def _combo_value(combo: QComboBox) -> str:
            value = combo.currentData()
            if isinstance(value, str):
                return value
            return combo.currentText().strip()

        @staticmethod
        def _format_size(size: object) -> str:
            try:
                value = int(size)
            except (TypeError, ValueError):
                return ""
            units = ("B", "KB", "MB", "GB", "TB")
            amount = float(value)
            for unit in units:
                if amount < 1024 or unit == units[-1]:
                    return f"{amount:.1f} {unit}" if unit != "B" else f"{value} B"
                amount /= 1024
            return str(value)

        def _scroll_tab(self) -> tuple[QWidget, QVBoxLayout]:
            page = QWidget()
            content = QWidget()
            layout = QVBoxLayout(content)
            layout.setContentsMargins(14, 14, 14, 14)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidget(content)
            root = QVBoxLayout(page)
            root.setContentsMargins(0, 0, 0, 0)
            root.addWidget(scroll)
            return page, layout

        def _build_storage_tab(self) -> QWidget:
            page, layout = self._scroll_tab()
            storage = QGroupBox("索引范围")
            form = QFormLayout(storage)
            self.db_path = QLineEdit()
            browse_db = QPushButton("选择位置")
            db_row = QWidget()
            db_layout = QHBoxLayout(db_row)
            db_layout.setContentsMargins(0, 0, 0, 0)
            db_layout.addWidget(self.db_path)
            db_layout.addWidget(browse_db)
            browse_db.clicked.connect(self.choose_database)
            self._form_widget(form, "索引数据库", db_row)

            self.folder_list = QListWidget()
            self.folder_list.setMinimumHeight(130)
            self._form_widget(form, "索引目录", self.folder_list)
            self.new_folder = QLineEdit()
            self.new_folder.setPlaceholderText("输入目录，或从文件选择器添加")
            add_folder = QPushButton("添加")
            browse_folder = QPushButton("选择目录")
            folder_row = QWidget()
            folder_layout = QHBoxLayout(folder_row)
            folder_layout.setContentsMargins(0, 0, 0, 0)
            folder_layout.addWidget(self.new_folder)
            folder_layout.addWidget(add_folder)
            folder_layout.addWidget(browse_folder)
            self._form_widget(form, "", folder_row)
            remove_folder = QPushButton("移除选中目录")
            self._form_widget(form, "", remove_folder)
            add_folder.clicked.connect(self.add_folder)
            browse_folder.clicked.connect(self.choose_folder)
            remove_folder.clicked.connect(self.remove_folder)

            self.exclude = QPlainTextEdit()
            self.exclude.setPlaceholderText("每行一个规则，例如 .git 或 *.app")
            self.exclude.setFixedHeight(80)
            self._form_widget(form, "排除规则", self.exclude)
            self.max_file_mb = self._int_box(0)
            self._form_widget(form, "单文件大小上限（MB）", self.max_file_mb)
            self.debounce = QDoubleSpinBox()
            self.debounce.setDecimals(1)
            self.debounce.setRange(0.1, 3600)
            self.debounce.setSingleStep(0.1)
            self._form_widget(form, "事件防抖（秒）", self.debounce)
            self.reconcile = self._int_box(0)
            self._form_widget(form, "全量对账间隔（秒）", self.reconcile)
            self.chunk_size = self._int_box(1)
            self._form_widget(form, "向量分块大小（字符）", self.chunk_size)
            self.chunk_overlap = self._int_box(0)
            self._form_widget(form, "向量分块重叠（字符）", self.chunk_overlap)
            layout.addWidget(storage)
            layout.addStretch()
            return page

        def _build_models_tab(self) -> QWidget:
            page, layout = self._scroll_tab()
            self.model_fields: dict[str, dict[str, Any]] = {}
            grid = QGridLayout()
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(12)
            for index, (name, label) in enumerate(self.MODEL_LABELS):
                box = QGroupBox(label)
                form = QFormLayout(box)
                enabled = QCheckBox("启用")
                mode = QComboBox()
                mode.addItem("OpenAI API", "openai")
                mode.addItem("本地模型", "local")
                local_model = QComboBox()
                local_model.addItem("未选择", "")
                base_url = QLineEdit()
                base_url.setPlaceholderText("https://api.openai.com/v1")
                model = QLineEdit()
                api_key = QLineEdit()
                api_key.setEchoMode(QLineEdit.EchoMode.Password)
                api_key.setPlaceholderText("留空以保留已保存的密钥")
                clear = QCheckBox("删除已保存密钥")
                self._form_widget(form, "", enabled)
                self._form_widget(form, "来源", mode)
                self._form_widget(form, "本地模型", local_model)
                self._form_widget(form, "接口地址", base_url)
                self._form_widget(form, "API 模型", model)
                self._form_widget(form, "API Key", api_key)
                self._form_widget(form, "", clear)
                self.model_fields[name] = {
                    "form": form,
                    "enabled": enabled,
                    "mode": mode,
                    "local_model": local_model,
                    "base_url": base_url,
                    "model": model,
                    "api_key": api_key,
                    "clear_api_key": clear,
                    "api_widgets": (base_url, model, api_key, clear),
                }
                mode.currentIndexChanged.connect(
                    lambda _index, purpose=name: self._sync_model_source_fields(purpose)
                )
                grid.addWidget(box, index // 2, index % 2)
            layout.addLayout(grid)

            manager_box = QGroupBox("本地模型管理")
            manager = QFormLayout(manager_box)
            self.local_model_dir = QLineEdit()
            self.local_model_dir.setReadOnly(True)
            open_model_dir = QPushButton("打开模型目录")
            model_dir_row = QWidget()
            model_dir_layout = QHBoxLayout(model_dir_row)
            model_dir_layout.setContentsMargins(0, 0, 0, 0)
            model_dir_layout.addWidget(self.local_model_dir)
            model_dir_layout.addWidget(open_model_dir)
            self._form_widget(manager, "模型目录", model_dir_row)
            self.local_runtime_status = QLabel()
            self.local_runtime_status.setWordWrap(True)
            self.local_runtime_status.setStyleSheet("color: palette(mid);")
            self._form_widget(manager, "运行时", self.local_runtime_status)

            self.local_models_tree = QTreeWidget()
            self.local_models_tree.setColumnCount(5)
            self.local_models_tree.setHeaderLabels(["模型", "格式", "大小", "能力", "状态"])
            self.local_models_tree.setRootIsDecorated(False)
            self.local_models_tree.setAlternatingRowColors(True)
            self.local_models_tree.setMinimumHeight(170)
            header = self.local_models_tree.header()
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            for column in range(1, 5):
                header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
            self._form_widget(manager, "可用模型", self.local_models_tree)

            self.local_model_capability = QComboBox()
            self.local_model_capability.addItem("聊天 / 文本", "chat")
            self.local_model_capability.addItem("向量", "embedding")
            self.local_model_capability.addItem("视觉", "vision")
            self.local_model_capability.addItem("语音识别", "asr")
            refresh_models = QPushButton("刷新")
            load_model = QPushButton("加载到内存")
            unload_model = QPushButton("卸载所选能力")
            unload_all = QPushButton("全部卸载")
            model_actions = QWidget()
            model_actions_layout = QHBoxLayout(model_actions)
            model_actions_layout.setContentsMargins(0, 0, 0, 0)
            model_actions_layout.addWidget(self.local_model_capability)
            model_actions_layout.addWidget(refresh_models)
            model_actions_layout.addWidget(load_model)
            model_actions_layout.addWidget(unload_model)
            model_actions_layout.addWidget(unload_all)
            self._form_widget(manager, "内存管理", model_actions)
            self._model_manager_buttons = (refresh_models, load_model, unload_model, unload_all)
            open_model_dir.clicked.connect(self.open_model_directory)
            refresh_models.clicked.connect(lambda: self.refresh_local_models(show_errors=True))
            load_model.clicked.connect(self.load_selected_local_model)
            unload_model.clicked.connect(lambda: self.unload_selected_local_model(all_capabilities=False))
            unload_all.clicked.connect(lambda: self.unload_selected_local_model(all_capabilities=True))
            layout.addWidget(manager_box)
            layout.addStretch()
            return page

        def _set_form_field_visible(self, form: QFormLayout, widget: QWidget, visible: bool) -> None:
            widget.setVisible(visible)
            label = form.labelForField(widget)
            if label is not None:
                label.setVisible(visible)

        def _sync_model_source_fields(self, purpose: str) -> None:
            fields = self.model_fields[purpose]
            local = self._combo_value(fields["mode"]) == "local"
            form = fields["form"]
            self._set_form_field_visible(form, fields["local_model"], local)
            for widget in fields["api_widgets"]:
                self._set_form_field_visible(form, widget, not local)

        def _sync_asr_provider_fields(self) -> None:
            local = self._combo_value(self.asr_provider) == "local"
            self._set_form_field_visible(self.asr_form, self.asr_local_model, local)
            self._set_form_field_visible(self.asr_form, self.asr_local_backend, local)
            for widget in self.asr_api_widgets:
                self._set_form_field_visible(self.asr_form, widget, not local)

        def _populate_local_model_combo(
            self,
            combo: QComboBox,
            value: str,
            capability: str,
        ) -> None:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("未选择", "")
            for model in self._model_catalog.get("models", []):
                if not isinstance(model, dict):
                    continue
                model_id = str(model.get("id", "")).strip()
                if not model_id:
                    continue
                capabilities = model.get("capabilities", [])
                if not isinstance(capabilities, list) or capability not in capabilities:
                    continue
                name = str(model.get("name", model_id))
                model_format = str(model.get("format", "")).upper()
                combo.addItem(f"{name} ({model_format})" if model_format else name, model_id)
            self._set_combo(combo, value)
            combo.blockSignals(False)

        def _selected_local_model(self) -> dict[str, Any] | None:
            item = self.local_models_tree.currentItem()
            if item is None:
                return None
            value = item.data(0, Qt.ItemDataRole.UserRole)
            return value if isinstance(value, dict) else None

        def _set_model_manager_busy(self, busy: bool) -> None:
            for button in self._model_manager_buttons:
                button.setEnabled(not busy)

        def _run_model_action(
            self,
            action: Callable[[], Any],
            completed: Callable[[Any, str | None], None],
        ) -> None:
            bridge = TaskBridge(self)
            self._model_bridges.append(bridge)
            self._set_model_manager_busy(True)

            def done(result: Any, error: Any) -> None:
                if bridge in self._model_bridges:
                    self._model_bridges.remove(bridge)
                self._set_model_manager_busy(False)
                completed(result, str(error) if error else None)

            bridge.finished.connect(done)

            def runner() -> None:
                try:
                    bridge.finished.emit(action(), None)
                except Exception as exc:
                    bridge.finished.emit(None, str(exc))

            threading.Thread(target=runner, daemon=True, name="semdex-model-manager").start()

        def refresh_local_models(self, *, show_errors: bool = False) -> None:
            try:
                catalog = self.controller.local_model_catalog()
            except Exception as exc:
                self.local_runtime_status.setText(f"无法读取本地模型：{exc}")
                if show_errors:
                    QMessageBox.warning(self, "无法刷新模型", str(exc))
                return
            self._model_catalog = catalog if isinstance(catalog, dict) else {}
            self.local_model_dir.setText(str(self._model_catalog.get("model_dir", "")))
            runtimes = self._model_catalog.get("runtimes", [])
            details: list[str] = []
            if isinstance(runtimes, list):
                for runtime in runtimes:
                    if not isinstance(runtime, dict):
                        continue
                    name = str(runtime.get("id", "运行时"))
                    state = "可用" if runtime.get("available") else "不可用"
                    detail = str(runtime.get("detail", "")).strip()
                    details.append(f"{name}: {state}" + (f"（{detail}）" if detail else ""))
            self.local_runtime_status.setText("  |  ".join(details) or "未检测到本地模型运行时")

            current_values = {
                name: self._combo_value(fields["local_model"])
                for name, fields in self.model_fields.items()
            }
            current_values["asr"] = self._combo_value(self.asr_local_model)
            for name, fields in self.model_fields.items():
                self._populate_local_model_combo(
                    fields["local_model"],
                    current_values[name],
                    self.MODEL_CAPABILITIES[name],
                )
            self._populate_local_model_combo(self.asr_local_model, current_values["asr"], "asr")

            self.local_models_tree.clear()
            models = self._model_catalog.get("models", [])
            if isinstance(models, list):
                for model in models:
                    if not isinstance(model, dict):
                        continue
                    model_id = str(model.get("id", "")).strip()
                    if not model_id:
                        continue
                    capabilities = model.get("capabilities", [])
                    if not isinstance(capabilities, list):
                        capabilities = []
                    loaded = model.get("loaded")
                    if isinstance(loaded, list):
                        loaded_text = "已加载: " + ", ".join(str(item) for item in loaded)
                    elif loaded:
                        loaded_text = "已加载"
                    elif model.get("loadable"):
                        loaded_text = "可加载"
                    else:
                        loaded_text = str(model.get("reason", "不可加载"))
                    item = QTreeWidgetItem([
                        str(model.get("name", model_id)),
                        str(model.get("format", "")).upper(),
                        self._format_size(model.get("size_bytes")),
                        ", ".join(str(item) for item in capabilities),
                        loaded_text,
                    ])
                    item.setToolTip(0, model_id)
                    reason = str(model.get("reason", "")).strip()
                    if reason:
                        item.setToolTip(4, reason)
                    item.setData(0, Qt.ItemDataRole.UserRole, model)
                    self.local_models_tree.addTopLevelItem(item)

        def open_model_directory(self) -> None:
            directory = self.local_model_dir.text().strip()
            if not directory:
                QMessageBox.warning(self, "无法打开目录", "模型目录尚不可用")
                return
            QDesktopServices.openUrl(QUrl.fromLocalFile(directory))

        def load_selected_local_model(self) -> None:
            model = self._selected_local_model()
            if model is None:
                QMessageBox.information(self, "选择模型", "请先在可用模型列表中选择一个模型")
                return
            model_id = str(model.get("id", ""))
            capability = self._combo_value(self.local_model_capability)

            def complete(result: Any, error: str | None) -> None:
                if error:
                    QMessageBox.warning(self, "无法加载模型", error)
                    return
                self.refresh_local_models()
                detail = str(result.get("detail", "已加载到内存")) if isinstance(result, dict) else "已加载到内存"
                QMessageBox.information(self, "Semdex", detail)

            self._run_model_action(
                lambda: self.controller.load_local_model(model_id, capability), complete
            )

        def unload_selected_local_model(self, *, all_capabilities: bool) -> None:
            model = self._selected_local_model()
            if model is None:
                QMessageBox.information(self, "选择模型", "请先在可用模型列表中选择一个模型")
                return
            model_id = str(model.get("id", ""))
            capability = None if all_capabilities else self._combo_value(self.local_model_capability)

            def complete(result: Any, error: str | None) -> None:
                if error:
                    QMessageBox.warning(self, "无法卸载模型", error)
                    return
                self.refresh_local_models()
                detail = str(result.get("detail", "模型已卸载")) if isinstance(result, dict) else "模型已卸载"
                QMessageBox.information(self, "Semdex", detail)

            self._run_model_action(
                lambda: self.controller.unload_local_model(model_id, capability), complete
            )

        def _build_tools_tab(self) -> QWidget:
            page, layout = self._scroll_tab()

            ocr_box = QGroupBox("OCR")
            ocr = QFormLayout(ocr_box)
            self.ocr_enabled = QCheckBox("启用")
            self.ocr_provider = QComboBox()
            self.ocr_provider.addItem("Tesseract（本地命令）", "tesseract")
            self.ocr_provider.addItem("本地 HTTP（PaddleOCR 等）", "local_http")
            self.ocr_command = QLineEdit()
            self.ocr_renderer = QLineEdit()
            self.ocr_languages = QLineEdit()
            self.ocr_dpi = self._int_box(72)
            self.ocr_endpoint = QLineEdit()
            self.ocr_response_path = QLineEdit()
            self.ocr_api_key = QLineEdit()
            self.ocr_api_key.setEchoMode(QLineEdit.EchoMode.Password)
            self.ocr_api_key.setPlaceholderText("留空以保留已保存的密钥")
            self.ocr_clear_api_key = QCheckBox("删除已保存密钥")
            self.ocr_timeout = self._int_box(1)
            for label, field in [
                ("", self.ocr_enabled), ("提供方", self.ocr_provider), ("命令", self.ocr_command),
                ("PDF 渲染器", self.ocr_renderer), ("语言", self.ocr_languages), ("DPI", self.ocr_dpi),
                ("接口地址", self.ocr_endpoint), ("响应文本路径", self.ocr_response_path),
                ("API Key", self.ocr_api_key), ("", self.ocr_clear_api_key), ("超时（秒）", self.ocr_timeout),
            ]:
                self._form_widget(ocr, label, field)

            asr_box = QGroupBox("语音识别")
            self.asr_form = QFormLayout(asr_box)
            self.asr_enabled = QCheckBox("启用")
            self.asr_provider = QComboBox()
            self.asr_provider.addItem("本地模型", "local")
            self.asr_provider.addItem("OpenAI API", "openai_compatible")
            self.asr_local_model = QComboBox()
            self.asr_local_model.addItem("未选择", "")
            self.asr_local_backend = QComboBox()
            self.asr_local_backend.addItem("自动", "auto")
            self.asr_local_backend.addItem("faster-whisper", "faster_whisper")
            self.asr_local_backend.addItem("whisper.cpp", "whisper_cpp")
            self.asr_local_backend.addItem("MLX Whisper", "mlx_whisper")
            self.asr_model = QLineEdit()
            self.asr_base_url = QLineEdit()
            self.asr_endpoint = QLineEdit()
            self.asr_language = QLineEdit()
            self.asr_response_path = QLineEdit()
            self.asr_api_key = QLineEdit()
            self.asr_api_key.setEchoMode(QLineEdit.EchoMode.Password)
            self.asr_api_key.setPlaceholderText("留空以保留已保存的密钥")
            self.asr_clear_api_key = QCheckBox("删除已保存密钥")
            self.asr_timeout = self._int_box(1)
            for label, field in [
                ("", self.asr_enabled), ("来源", self.asr_provider), ("本地模型", self.asr_local_model),
                ("本地后端", self.asr_local_backend), ("API 模型", self.asr_model),
                ("接口基地址", self.asr_base_url), ("转写接口", self.asr_endpoint),
                ("语言", self.asr_language), ("响应文本路径", self.asr_response_path),
                ("API Key", self.asr_api_key), ("", self.asr_clear_api_key), ("超时（秒）", self.asr_timeout),
            ]:
                self._form_widget(self.asr_form, label, field)
            self.asr_api_widgets = (
                self.asr_model,
                self.asr_base_url,
                self.asr_endpoint,
                self.asr_language,
                self.asr_response_path,
                self.asr_api_key,
                self.asr_clear_api_key,
                self.asr_timeout,
            )
            self.asr_provider.currentIndexChanged.connect(self._sync_asr_provider_fields)
            layout.addWidget(ocr_box)
            layout.addWidget(asr_box)
            layout.addStretch()
            return page

        def _build_features_tab(self) -> QWidget:
            page, layout = self._scroll_tab()
            box = QGroupBox("功能开关")
            form = QFormLayout(box)
            self.entities_enabled = QCheckBox("启用实体关系")
            self.entities_max_chars = self._int_box(500)
            self.entities_max_per_file = self._int_box(1)
            self.agent_max_steps = self._int_box(1)
            self.agent_max_results = self._int_box(1)
            self.fallback_enabled = QCheckBox("启用提取兜底")
            self.fallback_max_bytes = self._int_box(1024)
            for label, field in [
                ("", self.entities_enabled), ("实体最大文本长度", self.entities_max_chars),
                ("每文件最多实体", self.entities_max_per_file), ("Agent 最大步骤", self.agent_max_steps),
                ("Agent 最大结果数", self.agent_max_results), ("", self.fallback_enabled),
                ("兜底读取上限（字节）", self.fallback_max_bytes),
            ]:
                self._form_widget(form, label, field)
            layout.addWidget(box)
            layout.addStretch()
            return page

        def choose_database(self) -> None:
            path, _ = QFileDialog.getSaveFileName(self, "选择索引数据库", self.db_path.text(), "SQLite (*.db);;所有文件 (*)")
            if path:
                self.db_path.setText(path)

        def choose_folder(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "选择索引目录", self.new_folder.text() or str(Path.home()))
            if path:
                self.new_folder.setText(path)
                self.add_folder()

        def add_folder(self) -> None:
            path = self.new_folder.text().strip()
            if not path:
                return
            existing = {self.folder_list.item(index).text() for index in range(self.folder_list.count())}
            if path not in existing:
                self.folder_list.addItem(path)
            self.new_folder.clear()

        def remove_folder(self) -> None:
            for item in self.folder_list.selectedItems():
                self.folder_list.takeItem(self.folder_list.row(item))

        def load_settings(self, settings: dict[str, Any]) -> None:
            self.db_path.setText(str(settings.get("db_path", "")))
            self.folder_list.clear()
            for folder in settings.get("folders", []):
                self.folder_list.addItem(str(folder))
            self.exclude.setPlainText("\n".join(settings.get("exclude", [])))
            self.max_file_mb.setValue(int(settings.get("max_file_mb", 50)))
            self.debounce.setValue(float(settings.get("watch_debounce_sec", 1.5)))
            self.reconcile.setValue(int(settings.get("watch_reconcile_sec", 86400)))
            self.chunk_size.setValue(int(settings.get("chunk_size", 800)))
            self.chunk_overlap.setValue(int(settings.get("chunk_overlap", 100)))
            for name, fields in self.model_fields.items():
                model = settings.get("models", {}).get(name, {})
                fields["enabled"].setChecked(bool(model.get("enabled", False)))
                self._set_combo(fields["mode"], str(model.get("mode", "openai")))
                self._set_combo(fields["local_model"], str(model.get("local_model", "")))
                fields["base_url"].setText(str(model.get("base_url", "")))
                fields["model"].setText(str(model.get("model", "")))
                fields["api_key"].setText("")
                fields["clear_api_key"].setChecked(False)
                if model.get("api_key_configured"):
                    fields["api_key"].setPlaceholderText("已保存；留空以保留")
                self._sync_model_source_fields(name)

            ocr = settings.get("ocr", {})
            self.ocr_enabled.setChecked(bool(ocr.get("enabled", False)))
            self._set_combo(self.ocr_provider, str(ocr.get("provider", "tesseract")))
            self.ocr_command.setText(str(ocr.get("command", "tesseract")))
            self.ocr_renderer.setText(str(ocr.get("pdf_renderer", "pdftoppm")))
            self.ocr_languages.setText(str(ocr.get("languages", "eng+chi_sim")))
            self.ocr_dpi.setValue(int(ocr.get("dpi", 200)))
            self.ocr_endpoint.setText(str(ocr.get("endpoint", "")))
            self.ocr_response_path.setText(str(ocr.get("response_path", "text")))
            self.ocr_api_key.setText("")
            self.ocr_clear_api_key.setChecked(False)
            self.ocr_timeout.setValue(int(ocr.get("timeout_sec", 180)))
            if ocr.get("api_key_configured"):
                self.ocr_api_key.setPlaceholderText("已保存；留空以保留")

            asr = settings.get("asr", {})
            self.asr_enabled.setChecked(bool(asr.get("enabled", False)))
            provider = str(asr.get("provider", "local"))
            if provider == "faster_whisper":
                provider = "local"
            self._set_combo(self.asr_provider, provider)
            local_model = str(asr.get("local_model", ""))
            if not local_model and provider == "local":
                local_model = str(asr.get("model", ""))
            self._set_combo(self.asr_local_model, local_model)
            self._set_combo(self.asr_local_backend, str(asr.get("local_backend", "auto")))
            self.asr_model.setText(str(asr.get("model", "base")))
            self.asr_base_url.setText(str(asr.get("base_url", "")))
            self.asr_endpoint.setText(str(asr.get("endpoint", "")))
            self.asr_language.setText(str(asr.get("language", "")))
            self.asr_response_path.setText(str(asr.get("response_path", "text")))
            self.asr_api_key.setText("")
            self.asr_clear_api_key.setChecked(False)
            self.asr_timeout.setValue(int(asr.get("timeout_sec", 180)))
            if asr.get("api_key_configured"):
                self.asr_api_key.setPlaceholderText("已保存；留空以保留")
            self._sync_asr_provider_fields()

            entities = settings.get("entities", {})
            agent = settings.get("agent", {})
            fallback = settings.get("agent_fallback", {})
            self.entities_enabled.setChecked(bool(entities.get("enabled", False)))
            self.entities_max_chars.setValue(int(entities.get("max_chars", 12000)))
            self.entities_max_per_file.setValue(int(entities.get("max_per_file", 32)))
            self.agent_max_steps.setValue(int(agent.get("max_steps", 6)))
            self.agent_max_results.setValue(int(agent.get("max_results", 12)))
            self.fallback_enabled.setChecked(bool(fallback.get("enabled", False)))
            self.fallback_max_bytes.setValue(int(fallback.get("max_bytes", 262144)))

        def payload(self) -> dict[str, Any]:
            if self.chunk_overlap.value() >= self.chunk_size.value():
                raise ValueError("向量分块重叠必须小于分块大小")
            folders = [self.folder_list.item(index).text() for index in range(self.folder_list.count())]
            models: dict[str, dict[str, Any]] = {}
            for name, fields in self.model_fields.items():
                item = {
                    "enabled": fields["enabled"].isChecked(),
                    "mode": self._combo_value(fields["mode"]),
                    "local_model": self._combo_value(fields["local_model"]),
                    "base_url": fields["base_url"].text().strip(),
                    "model": fields["model"].text().strip(),
                }
                key = fields["api_key"].text().strip()
                if key:
                    item["api_key"] = key
                if fields["clear_api_key"].isChecked():
                    item["clear_api_key"] = True
                models[name] = item
            ocr: dict[str, Any] = {
                "enabled": self.ocr_enabled.isChecked(),
                "provider": self.ocr_provider.currentData(),
                "command": self.ocr_command.text().strip(),
                "pdf_renderer": self.ocr_renderer.text().strip(),
                "languages": self.ocr_languages.text().strip(),
                "dpi": self.ocr_dpi.value(),
                "endpoint": self.ocr_endpoint.text().strip(),
                "response_path": self.ocr_response_path.text().strip(),
                "timeout_sec": self.ocr_timeout.value(),
            }
            ocr_key = self.ocr_api_key.text().strip()
            if ocr_key:
                ocr["api_key"] = ocr_key
            if self.ocr_clear_api_key.isChecked():
                ocr["clear_api_key"] = True
            asr: dict[str, Any] = {
                "enabled": self.asr_enabled.isChecked(),
                "provider": self._combo_value(self.asr_provider),
                "local_model": self._combo_value(self.asr_local_model),
                "local_backend": self._combo_value(self.asr_local_backend),
                "model": self.asr_model.text().strip(),
                "base_url": self.asr_base_url.text().strip(),
                "endpoint": self.asr_endpoint.text().strip(),
                "language": self.asr_language.text().strip(),
                "response_path": self.asr_response_path.text().strip(),
                "timeout_sec": self.asr_timeout.value(),
            }
            asr_key = self.asr_api_key.text().strip()
            if asr_key:
                asr["api_key"] = asr_key
            if self.asr_clear_api_key.isChecked():
                asr["clear_api_key"] = True
            return {
                "db_path": self.db_path.text().strip(),
                "folders": folders,
                "exclude": [line.strip() for line in self.exclude.toPlainText().splitlines() if line.strip()],
                "max_file_mb": self.max_file_mb.value(),
                "watch_debounce_sec": self.debounce.value(),
                "watch_reconcile_sec": self.reconcile.value(),
                "chunk_size": self.chunk_size.value(),
                "chunk_overlap": self.chunk_overlap.value(),
                "models": models,
                "ocr": ocr,
                "asr": asr,
                "entities": {
                    "enabled": self.entities_enabled.isChecked(),
                    "max_chars": self.entities_max_chars.value(),
                    "max_per_file": self.entities_max_per_file.value(),
                },
                "agent": {
                    "max_steps": self.agent_max_steps.value(),
                    "max_results": self.agent_max_results.value(),
                },
                "agent_fallback": {
                    "enabled": self.fallback_enabled.isChecked(),
                    "max_bytes": self.fallback_max_bytes.value(),
                },
            }

        def save(self) -> bool:
            try:
                self.controller.save_settings(self.payload())
            except Exception as exc:
                QMessageBox.warning(self, "无法保存设置", str(exc))
                return False
            QMessageBox.information(self, "Semdex", "设置已保存")
            return True

        def save_and_index(self) -> None:
            if self.save():
                self.start_after_save = True
                self.accept()

    class MainWindow(QMainWindow):
        def __init__(self, controller: DesktopController):
            super().__init__()
            self.controller = controller
            self._bridges: list[TaskBridge] = []
            self._search_requests = SearchRequestGuard()
            self.setWindowTitle("Semdex")
            self.resize(1080, 760)
            self._build_ui()
            self._search_timer = QTimer(self)
            self._search_timer.setSingleShot(True)
            self._search_timer.setInterval(280)
            self._search_timer.timeout.connect(self.perform_search)
            self._status_timer = QTimer(self)
            self._status_timer.timeout.connect(self.refresh_status)
            self._status_timer.start(1200)
            self.refresh_status()

        def _build_ui(self) -> None:
            settings_action = QAction("设置", self)
            settings_action.triggered.connect(self.show_settings)
            reindex_action = QAction("重新索引", self)
            reindex_action.triggered.connect(lambda: self.start_index(False))
            rebuild_action = QAction("完整重建", self)
            rebuild_action.triggered.connect(lambda: self.start_index(True))
            toolbar = self.addToolBar("操作")
            toolbar.setMovable(False)
            toolbar.addAction(settings_action)
            toolbar.addSeparator()
            toolbar.addAction(reindex_action)
            toolbar.addAction(rebuild_action)

            content = QWidget()
            layout = QVBoxLayout(content)
            layout.setContentsMargins(22, 18, 22, 18)
            title_row = QHBoxLayout()
            title = QLabel("Semdex")
            title.setStyleSheet("font-size: 23px; font-weight: 650;")
            subtitle = QLabel("本地文件语义索引")
            subtitle.setStyleSheet("color: palette(mid);")
            title_row.addWidget(title)
            title_row.addWidget(subtitle)
            title_row.addStretch()
            layout.addLayout(title_row)

            self.status_label = QLabel("正在读取索引状态…")
            self.status_label.setWordWrap(True)
            self.status_label.setStyleSheet("padding: 7px 0; color: palette(mid);")
            layout.addWidget(self.status_label)

            search_row = QHBoxLayout()
            self.query = QLineEdit()
            self.query.setPlaceholderText("搜索文件名或内容，回车或停顿即搜…")
            self.query.setClearButtonEnabled(True)
            self.query.returnPressed.connect(self.perform_search)
            self.query.textChanged.connect(lambda: self._search_timer.start())
            search_row.addWidget(self.query)
            layout.addLayout(search_row)

            modes = QHBoxLayout()
            self.mode_group = QButtonGroup(self)
            self.mode_group.setExclusive(True)
            for index, (mode, label) in enumerate([
                ("hybrid", "混合"), ("fulltext", "关键词"), ("semantic", "语义"), ("ask", "问答"),
            ]):
                button = QPushButton(label)
                button.setCheckable(True)
                button.setChecked(index == 0)
                button.setProperty("mode", mode)
                button.clicked.connect(self.perform_search)
                self.mode_group.addButton(button)
                modes.addWidget(button)
            modes.addStretch()
            layout.addLayout(modes)

            self.answer = QPlainTextEdit()
            self.answer.setReadOnly(True)
            self.answer.setMaximumHeight(115)
            self.answer.setPlaceholderText("问答结果会显示在这里")
            self.answer.hide()
            layout.addWidget(self.answer)

            self.results = QTreeWidget()
            self.results.setColumnCount(4)
            self.results.setHeaderLabels(["文件", "路径", "摘要", "来源"])
            self.results.setRootIsDecorated(False)
            self.results.setAlternatingRowColors(True)
            self.results.itemSelectionChanged.connect(self.update_actions)
            self.results.itemDoubleClicked.connect(lambda *_: self.show_content())
            header = self.results.header()
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            layout.addWidget(self.results, 1)

            actions = QHBoxLayout()
            self.open_button = QPushButton("打开")
            self.reveal_button = QPushButton("在文件管理器中显示")
            self.content_button = QPushButton("查看全文")
            for button in (self.open_button, self.reveal_button, self.content_button):
                button.setEnabled(False)
                actions.addWidget(button)
            actions.addStretch()
            self.open_button.clicked.connect(lambda: self.open_selected(False))
            self.reveal_button.clicked.connect(lambda: self.open_selected(True))
            self.content_button.clicked.connect(self.show_content)
            layout.addLayout(actions)
            self.setCentralWidget(content)

        def _run_async(self, func: Callable[[], Any], callback: Callable[[Any, str | None], None]) -> None:
            bridge = TaskBridge(self)
            self._bridges.append(bridge)

            def done(result: Any, error: Any) -> None:
                if bridge in self._bridges:
                    self._bridges.remove(bridge)
                callback(result, str(error) if error else None)

            bridge.finished.connect(done)

            def runner() -> None:
                try:
                    bridge.finished.emit(func(), None)
                except Exception as exc:
                    bridge.finished.emit(None, str(exc))

            threading.Thread(target=runner, daemon=True, name="semdex-gui-task").start()

        def selected_mode(self) -> str:
            button = self.mode_group.checkedButton()
            return str(button.property("mode")) if button else "hybrid"

        def refresh_status(self) -> None:
            try:
                status = self.controller.status()
            except Exception as exc:
                self.status_label.setText(f"无法读取索引状态：{exc}")
                return
            files = status["files"]
            by_status = files.get("by_status", {})
            pieces = [f"文件 {files.get('total', 0)}", f"已索引 {by_status.get('done', 0)}"]
            for key, label in [
                ("pending", "待索引"), ("waiting_model", "等待模型"),
                ("waiting_capability", "等待本地能力"), ("too_large", "超过大小限制"), ("failed", "失败"),
            ]:
                if by_status.get(key):
                    pieces.append(f"{label} {by_status[key]}")
            semantic = "需重建" if status["embedding_rebuild_required"] else ("开" if status["models"]["embedding"] else "关")
            pieces.append(f"语义搜索 {semantic}")
            pieces.append(f"图片识别 {'开' if status['models']['vision'] else '关'}")
            if files.get("entities"):
                pieces.append(f"实体 {files['entities']}")
            if status["indexing"]:
                pieces.append("正在索引…")
                self._status_timer.setInterval(800)
            else:
                self._status_timer.setInterval(1800)
                last_run = status.get("last_run") or {}
                if last_run.get("error"):
                    pieces.append(f"上次索引失败：{last_run['error']}")
            self.status_label.setText("  |  ".join(pieces))

        def perform_search(self) -> None:
            generation = self._search_requests.begin()
            query = self.query.text().strip()
            if not query:
                self.answer.hide()
                self.results.clear()
                self.update_actions()
                return
            mode = self.selected_mode()
            self.statusBar().showMessage("正在搜索…")

            def complete(result: Any, error: str | None) -> None:
                if not self._search_requests.is_current(generation):
                    return
                if error:
                    self.statusBar().showMessage(error, 6000)
                    return
                self.results.clear()
                if mode == "ask":
                    self.answer.setPlainText(str(result.get("answer", "")))
                    self.answer.show()
                    hits = result.get("hits", [])
                else:
                    self.answer.hide()
                    hits = result
                for hit in hits:
                    item = QTreeWidgetItem([
                        str(hit.get("filename", "")),
                        str(hit.get("path", "")),
                        str(hit.get("snippet", "")),
                        str(hit.get("source", "")),
                    ])
                    item.setData(0, Qt.ItemDataRole.UserRole, hit)
                    self.results.addTopLevelItem(item)
                self.statusBar().showMessage(f"找到 {len(hits)} 个结果", 3000)
                self.update_actions()

            if mode == "ask":
                self._run_async(lambda: self.controller.ask(query), complete)
            else:
                self._run_async(lambda: self.controller.search(query, mode), complete)

        def active_hit(self) -> dict[str, Any] | None:
            item = self.results.currentItem()
            if item is None:
                return None
            value = item.data(0, Qt.ItemDataRole.UserRole)
            return value if isinstance(value, dict) else None

        def update_actions(self) -> None:
            enabled = self.active_hit() is not None
            self.open_button.setEnabled(enabled)
            self.reveal_button.setEnabled(enabled)
            self.content_button.setEnabled(enabled)

        def open_selected(self, reveal: bool) -> None:
            hit = self.active_hit()
            if hit is None:
                return
            try:
                self.controller.open_path(str(hit["path"]), reveal=reveal)
            except Exception as exc:
                QMessageBox.warning(self, "无法打开文件", str(exc))

        def show_content(self) -> None:
            hit = self.active_hit()
            if hit is None:
                return
            self.statusBar().showMessage("正在读取全文…")

            def complete(result: Any, error: str | None) -> None:
                if error:
                    QMessageBox.warning(self, "无法读取全文", error)
                    return
                dialog = QDialog(self)
                dialog.setWindowTitle(str(result.get("path", "全文")))
                dialog.resize(800, 620)
                layout = QVBoxLayout(dialog)
                metadata = QLabel(f"提取器：{result.get('extractor') or '未知'}")
                metadata.setStyleSheet("color: palette(mid);")
                layout.addWidget(metadata)
                text = QPlainTextEdit()
                text.setReadOnly(True)
                text.setPlainText(str(result.get("text", "")))
                layout.addWidget(text, 1)
                entities = result.get("entities") or []
                if entities:
                    entity_text = ", ".join(str(entity.get("name", "")) for entity in entities)
                    layout.addWidget(QLabel(f"实体：{entity_text}"))
                close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
                close.rejected.connect(dialog.reject)
                layout.addWidget(close)
                dialog.exec()

            self._run_async(lambda: self.controller.content(int(hit["file_id"])), complete)

        def start_index(self, full_rebuild: bool) -> None:
            if full_rebuild:
                answer = QMessageBox.question(
                    self,
                    "完整重建",
                    "完整重建会重新提取全部已索引文件，可能需要较长时间。继续吗？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
            try:
                self.controller.start_index(full_rebuild=full_rebuild)
            except Exception as exc:
                QMessageBox.warning(self, "无法开始索引", str(exc))
                return
            self.refresh_status()

        def show_settings(self) -> None:
            dialog = SettingsDialog(self.controller, self)
            if dialog.exec() and dialog.start_after_save:
                self.start_index(False)
            self.refresh_status()

    try:
        controller = DesktopController(config_path)
    except Exception as exc:
        print(f"无法初始化 Semdex: {exc}", file=sys.stderr)
        return 1
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Semdex")
    window = MainWindow(controller)
    window.show()
    return app.exec()
