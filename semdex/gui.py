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
from .imagetypes import SUPPORTED_IMAGE_EXTENSIONS


def _normalize_extractor_input_mode(extensions_text: str, requested: str) -> str:
    """Keep raw-image input limited to rules containing only image extensions."""
    extensions = [
        value if value.startswith(".") else f".{value}"
        for value in (
            part.strip().lower() for part in extensions_text.split(",")
        )
        if value
    ]
    image_allowed = bool(extensions) and all(
        extension in SUPPORTED_IMAGE_EXTENSIONS for extension in extensions
    )
    return "image" if requested == "image" and image_allowed else "text"


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
            QProgressBar,
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
            ("agent", "检索 Agent"),
            ("entities", "实体抽取"),
            ("embedding", "语义嵌入"),
        ]
        MODEL_CAPABILITIES = {
            "agent": "chat",
            "entities": "chat",
            "embedding": "embedding",
        }
        MODEL_DESCRIPTIONS = {
            "agent": (
                "用于自然语言搜索与问答。它会调用全文、语义、实体和受限文件查看工具，整理结果但不会修改文件。"
            ),
            "entities": (
                "在一级正文索引完成后识别人名、机构、项目和日期等实体，并建立文件间的实体关系。"
            ),
            "embedding": (
                "在一级正文索引上分块生成向量，供语义和混合检索使用；更换模型后必须重建向量库。"
            ),
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
            tabs.addTab(self._build_extractors_tab(), "一级索引")
            tabs.addTab(self._build_models_tab(), "模型与二级索引")
            tabs.addTab(self._build_tools_tab(), "随附插件参数")
            tabs.addTab(self._build_features_tab(), "二级索引开关")
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

            providers_box = QGroupBox("LLM 供应商（一级索引）")
            providers_layout = QVBoxLayout(providers_box)
            providers_intro = QLabel(
                "这里的供应商只供扩展名规则的“传入 LLM”方式选择。可同时配置本地模型和云端 / "
                "OpenAI 兼容接口；初始的“默认 LLM”也可改名或删除。"
            )
            providers_intro.setWordWrap(True)
            providers_intro.setStyleSheet("color: palette(mid);")
            providers_layout.addWidget(providers_intro)
            self.provider_tree = QTreeWidget()
            self.provider_tree.setColumnCount(4)
            self.provider_tree.setHeaderLabels(["启用", "名称", "来源", "模型"])
            self.provider_tree.setRootIsDecorated(False)
            self.provider_tree.setAlternatingRowColors(True)
            self.provider_tree.setMinimumHeight(150)
            provider_header = self.provider_tree.header()
            provider_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            provider_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            for column in (2, 3):
                provider_header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
            providers_layout.addWidget(self.provider_tree)
            provider_actions = QHBoxLayout()
            add_provider = QPushButton("添加供应商")
            remove_provider = QPushButton("删除所选供应商")
            provider_actions.addWidget(add_provider)
            provider_actions.addWidget(remove_provider)
            provider_actions.addStretch()
            providers_layout.addLayout(provider_actions)

            self.provider_editor = QGroupBox("所选供应商")
            self.provider_editor_form = QFormLayout(self.provider_editor)
            self.provider_id = QLineEdit()
            self.provider_id.setReadOnly(True)
            self.provider_name = QLineEdit()
            self.provider_enabled = QCheckBox("启用")
            self.provider_mode = QComboBox()
            self.provider_mode.addItem("云端 / OpenAI 兼容接口", "openai")
            self.provider_mode.addItem("本地模型", "local")
            self.provider_local_model = QComboBox()
            self.provider_local_model.addItem("未选择", "")
            self.provider_base_url = QLineEdit()
            self.provider_base_url.setPlaceholderText("https://api.openai.com/v1")
            self.provider_model = QLineEdit()
            self.provider_api_key = QLineEdit()
            self.provider_api_key.setEchoMode(QLineEdit.EchoMode.Password)
            self.provider_api_key.setPlaceholderText("留空以保留已保存的密钥")
            self.provider_clear_api_key = QCheckBox("删除已保存密钥")
            for label, field in [
                ("稳定 ID", self.provider_id),
                ("显示名称", self.provider_name),
                ("", self.provider_enabled),
                ("来源", self.provider_mode),
                ("本地模型", self.provider_local_model),
                ("接口地址", self.provider_base_url),
                ("API 模型", self.provider_model),
                ("API Key", self.provider_api_key),
                ("", self.provider_clear_api_key),
            ]:
                self._form_widget(self.provider_editor_form, label, field)
            self.provider_editor.setEnabled(False)
            providers_layout.addWidget(self.provider_editor)
            self._provider_editor_loading = False
            self._provider_editor_item: QTreeWidgetItem | None = None
            add_provider.clicked.connect(self.add_llm_provider)
            remove_provider.clicked.connect(self.remove_llm_provider)
            self.provider_tree.currentItemChanged.connect(self.load_llm_provider_editor)
            self.provider_tree.itemChanged.connect(self.sync_llm_provider_tree_enabled)
            self.provider_name.textChanged.connect(self.apply_llm_provider_editor)
            self.provider_enabled.toggled.connect(self.apply_llm_provider_editor)
            self.provider_mode.currentIndexChanged.connect(self.apply_llm_provider_editor)
            self.provider_local_model.currentIndexChanged.connect(self.apply_llm_provider_editor)
            self.provider_base_url.textChanged.connect(self.apply_llm_provider_editor)
            self.provider_model.textChanged.connect(self.apply_llm_provider_editor)
            self.provider_api_key.textChanged.connect(self.apply_llm_provider_editor)
            self.provider_clear_api_key.toggled.connect(self.apply_llm_provider_editor)
            layout.addWidget(providers_box)

            secondary_title = QLabel("独立模型（检索与二级索引）")
            secondary_title.setStyleSheet("font-size: 14px; font-weight: 600;")
            layout.addWidget(secondary_title)
            self.model_fields: dict[str, dict[str, Any]] = {}
            grid = QGridLayout()
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(12)
            for index, (name, label) in enumerate(self.MODEL_LABELS):
                box = QGroupBox(label)
                form = QFormLayout(box)
                description = QLabel(self.MODEL_DESCRIPTIONS[name])
                description.setWordWrap(True)
                description.setStyleSheet("font-size: 11px;")
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
                form.addRow(description)
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
            embedding_guide = QLabel(
                "语义嵌入模型可从 <a href='https://huggingface.co/models?pipeline_tag=feature-extraction'>"
                "Hugging Face</a> 或 <a href='https://modelscope.cn/models'>ModelScope</a> 下载。"
                "GGUF 直接放入此目录；MLX 模型放入独立子文件夹，至少包含 config.json 和一个 "
                ".safetensors（或 consolidated.*）权重文件。目录名会成为本地模型 ID，并应含 embedding、"
                "bge、e5、nomic、gte 或 jina 之一（或让 config.json 的模型信息包含这些标识），才能识别为向量模型；"
                "同时保留该模型的 tokenizer 文件。选择后可在下方按用途提前加载或卸载，未提前加载时会在首次调用时按需加载。"
            )
            embedding_guide.setWordWrap(True)
            embedding_guide.setOpenExternalLinks(True)
            embedding_guide.setStyleSheet("font-size: 11px; color: palette(mid);")
            self._form_widget(manager, "嵌入模型添加", embedding_guide)
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

        def add_llm_provider_item(self, provider: dict[str, Any]) -> QTreeWidgetItem:
            provider_id = str(provider.get("id", "")).strip()
            mode = str(provider.get("mode", "openai")).strip().lower()
            if mode not in {"openai", "local"}:
                mode = "openai"
            metadata = {
                "id": provider_id,
                "mode": mode,
                "local_model": str(provider.get("local_model", "")),
                "base_url": str(provider.get("base_url", "")),
                "model": str(provider.get("model", "")),
                "api_key": "",
                "api_key_configured": bool(provider.get("api_key_configured", False)),
                "clear_api_key": False,
            }
            name = str(provider.get("name", provider_id)).strip() or provider_id
            model_label = metadata["local_model"] if mode == "local" else metadata["model"]
            item = QTreeWidgetItem([
                "",
                name,
                "本地" if mode == "local" else "API",
                model_label or "未设置",
            ])
            item.setCheckState(
                0,
                Qt.CheckState.Checked if provider.get("enabled", False) else Qt.CheckState.Unchecked,
            )
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setData(0, Qt.ItemDataRole.UserRole, metadata)
            self.provider_tree.addTopLevelItem(item)
            return item

        def add_llm_provider(self) -> None:
            existing_ids = {
                str((self.provider_tree.topLevelItem(index).data(0, Qt.ItemDataRole.UserRole) or {}).get("id", ""))
                for index in range(self.provider_tree.topLevelItemCount())
            }
            index = 1
            while f"provider-{index}" in existing_ids:
                index += 1
            item = self.add_llm_provider_item({
                "id": f"provider-{index}",
                "name": f"LLM 供应商 {index}",
                "enabled": False,
                "mode": "openai",
                "base_url": "http://localhost:1234/v1",
                "model": "",
            })
            self.provider_tree.setCurrentItem(item)
            self.refresh_extractor_provider_choices()

        def remove_llm_provider(self) -> None:
            item = self.provider_tree.currentItem()
            if item is None:
                return
            metadata = item.data(0, Qt.ItemDataRole.UserRole) or {}
            removed_id = str(metadata.get("id", ""))
            referenced_rules: list[str] = []
            for index in range(self.extractor_tree.topLevelItemCount()):
                rule_item = self.extractor_tree.topLevelItem(index)
                rule = rule_item.data(0, Qt.ItemDataRole.UserRole) or {}
                if rule.get("kind") == "llm" and rule.get("provider") == removed_id:
                    referenced_rules.append(rule_item.text(1) or str(rule.get("id", "")))
            if referenced_rules:
                QMessageBox.information(
                    self,
                    "LLM 供应商正在使用",
                    "请先为这些扩展名规则改选 LLM：" + "、".join(referenced_rules),
                )
                return
            self.provider_tree.takeTopLevelItem(self.provider_tree.indexOfTopLevelItem(item))
            self._provider_editor_item = None
            self.provider_editor.setEnabled(False)
            self.refresh_extractor_provider_choices()

        def _sync_provider_source_fields(self) -> None:
            local = self._combo_value(self.provider_mode) == "local"
            self._set_form_field_visible(
                self.provider_editor_form, self.provider_local_model, local
            )
            for widget in (
                self.provider_base_url,
                self.provider_model,
                self.provider_api_key,
                self.provider_clear_api_key,
            ):
                self._set_form_field_visible(self.provider_editor_form, widget, not local)

        def load_llm_provider_editor(
            self,
            item: QTreeWidgetItem | None,
            _previous: QTreeWidgetItem | None = None,
        ) -> None:
            self._provider_editor_loading = True
            self._provider_editor_item = item
            if item is None:
                self.provider_editor.setEnabled(False)
                self._provider_editor_loading = False
                return
            metadata = dict(item.data(0, Qt.ItemDataRole.UserRole) or {})
            self.provider_editor.setEnabled(True)
            self.provider_id.setText(str(metadata.get("id", "")))
            self.provider_name.setText(item.text(1))
            self.provider_enabled.setChecked(item.checkState(0) == Qt.CheckState.Checked)
            self._set_combo(self.provider_mode, str(metadata.get("mode", "openai")))
            self._set_combo(self.provider_local_model, str(metadata.get("local_model", "")))
            self.provider_base_url.setText(str(metadata.get("base_url", "")))
            self.provider_model.setText(str(metadata.get("model", "")))
            self.provider_api_key.setText(str(metadata.get("api_key", "")))
            self.provider_clear_api_key.setChecked(bool(metadata.get("clear_api_key", False)))
            self.provider_api_key.setPlaceholderText(
                "已保存；留空以保留"
                if metadata.get("api_key_configured") else "留空表示不设置"
            )
            self._sync_provider_source_fields()
            self._provider_editor_loading = False

        def apply_llm_provider_editor(self, *_args: Any) -> None:
            if self._provider_editor_loading or self._provider_editor_item is None:
                return
            item = self._provider_editor_item
            metadata = dict(item.data(0, Qt.ItemDataRole.UserRole) or {})
            mode = self._combo_value(self.provider_mode)
            metadata.update({
                "mode": mode,
                "local_model": self._combo_value(self.provider_local_model),
                "base_url": self.provider_base_url.text().strip(),
                "model": self.provider_model.text().strip(),
                "api_key": self.provider_api_key.text().strip(),
                "clear_api_key": self.provider_clear_api_key.isChecked(),
            })
            name = self.provider_name.text().strip() or str(metadata.get("id", ""))
            item.setText(1, name)
            item.setText(2, "本地" if mode == "local" else "API")
            selected_model = metadata["local_model"] if mode == "local" else metadata["model"]
            item.setText(3, selected_model or "未设置")
            item.setCheckState(
                0,
                Qt.CheckState.Checked if self.provider_enabled.isChecked() else Qt.CheckState.Unchecked,
            )
            item.setData(0, Qt.ItemDataRole.UserRole, metadata)
            self._sync_provider_source_fields()
            self.refresh_extractor_provider_choices()

        def sync_llm_provider_tree_enabled(
            self,
            item: QTreeWidgetItem,
            column: int,
        ) -> None:
            if (
                self._provider_editor_loading
                or column != 0
                or item is not self._provider_editor_item
            ):
                return
            checked = item.checkState(0) == Qt.CheckState.Checked
            if self.provider_enabled.isChecked() == checked:
                return
            self._provider_editor_loading = True
            self.provider_enabled.setChecked(checked)
            self._provider_editor_loading = False

        def refresh_extractor_provider_choices(self) -> None:
            current = self._combo_value(self.extractor_provider)
            providers: list[tuple[str, str]] = []
            for index in range(self.provider_tree.topLevelItemCount()):
                item = self.provider_tree.topLevelItem(index)
                metadata = item.data(0, Qt.ItemDataRole.UserRole) or {}
                provider_id = str(metadata.get("id", ""))
                if provider_id:
                    providers.append((provider_id, item.text(1) or provider_id))
            self.extractor_provider.blockSignals(True)
            self.extractor_provider.clear()
            self.extractor_provider.addItem("未选择", "")
            for provider_id, name in providers:
                self.extractor_provider.addItem(name, provider_id)
            self._set_combo(self.extractor_provider, current)
            self.extractor_provider.blockSignals(False)
            for index in range(self.extractor_tree.topLevelItemCount()):
                item = self.extractor_tree.topLevelItem(index)
                metadata = item.data(0, Qt.ItemDataRole.UserRole) or {}
                if metadata.get("kind") == "llm":
                    item.setText(4, self._extractor_settings_summary(metadata))

        def _build_extractors_tab(self) -> QWidget:
            page, layout = self._scroll_tab()
            box = QGroupBox("扩展名与索引方式")
            box_layout = QVBoxLayout(box)
            intro = QLabel(
                "这里建立一级正文索引。每条扩展名规则只使用三种方式之一：直接索引文本、传入 LLM、"
                "Python 外置插件。语义向量和实体关系会在这份正文库之上继续建立。"
            )
            intro.setWordWrap(True)
            intro.setStyleSheet("color: palette(mid);")
            box_layout.addWidget(intro)
            self.extractor_dir = QLineEdit()
            self.extractor_dir.setReadOnly(True)
            open_directory = QPushButton("打开目录")
            refresh_scripts = QPushButton("刷新插件列表")
            directory_row = QWidget()
            directory_layout = QHBoxLayout(directory_row)
            directory_layout.setContentsMargins(0, 0, 0, 0)
            directory_layout.addWidget(QLabel("Python 插件目录"))
            directory_layout.addWidget(self.extractor_dir, 1)
            directory_layout.addWidget(open_directory)
            directory_layout.addWidget(refresh_scripts)
            box_layout.addWidget(directory_row)

            guide = QGroupBox("Python 外置插件设置指引")
            guide_layout = QVBoxLayout(guide)
            guide_text = QLabel(
                "每个插件使用一个独立文件夹，入口固定为 plugin.py。函数接收安全快照路径（可选第二个 ctx 参数），"
                "返回字符串、字节串或其他可转成字符串的值作为一级索引正文。插件会执行本地 Python 代码，"
                "只放入可信插件。OCR 与语音识别也是同一种插件。"
            )
            guide_text.setWordWrap(True)
            guide_text.setStyleSheet("font-size: 11px;")
            guide_example = QPlainTextEdit()
            guide_example.setReadOnly(True)
            guide_example.setPlainText(
                "extractors/my_plugin/plugin.py\n\n"
                "from pathlib import Path\n\n"
                "def extract(path: Path) -> str:\n"
                "    return path.read_text(encoding='utf-8', errors='replace')"
            )
            guide_example.setFixedHeight(112)
            guide_layout.addWidget(guide_text)
            guide_layout.addWidget(guide_example)
            box_layout.addWidget(guide)

            self.extractor_tree = QTreeWidget()
            self.extractor_tree.setColumnCount(5)
            self.extractor_tree.setHeaderLabels(["启用", "规则", "扩展名（逗号分隔）", "索引方式", "当前设置"])
            self.extractor_tree.setRootIsDecorated(False)
            self.extractor_tree.setAlternatingRowColors(True)
            self.extractor_tree.setMinimumHeight(240)
            header = self.extractor_tree.header()
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
            for column in (3, 4):
                header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
            box_layout.addWidget(self.extractor_tree)
            actions = QHBoxLayout()
            add = QPushButton("添加自定义规则")
            remove = QPushButton("删除所选自定义规则")
            actions.addWidget(add)
            actions.addWidget(remove)
            actions.addStretch()
            box_layout.addLayout(actions)

            self.extractor_editor = QGroupBox("所选规则")
            self.extractor_editor_form = QFormLayout(self.extractor_editor)
            self.extractor_label = QLineEdit()
            self.extractor_extensions = QLineEdit()
            self.extractor_extensions.setPlaceholderText(".abc, .xyz")
            self.extractor_kind = QComboBox()
            self.extractor_kind.addItem("直接索引文本", "text")
            self.extractor_kind.addItem("传入 LLM", "llm")
            self.extractor_kind.addItem("Python 外置插件", "python")
            self.extractor_provider = QComboBox()
            self.extractor_provider.addItem("未选择", "")
            self.extractor_input_mode = QComboBox()
            self.extractor_input_mode.addItem("先提取为文本后传入", "text")
            self.extractor_input_mode.addItem("作为图片传入多模态模型", "image")
            self.extractor_prompt = QPlainTextEdit()
            self.extractor_prompt.setPlaceholderText("例如：提取与检索有关的事实、标题和关键词，保留原文中的名称与日期。")
            self.extractor_prompt.setFixedHeight(76)
            self.extractor_plugin = QComboBox()
            self.extractor_plugin.setEditable(True)
            self.extractor_plugin.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            self.extractor_plugin.setToolTip("填写插件目录中的单个文件夹名；旧版 .py 文件名仍可兼容")
            self.extractor_function = QLineEdit()
            self.extractor_function.setPlaceholderText("extract")
            self.extractor_kind_hint = QLabel()
            self.extractor_kind_hint.setWordWrap(True)
            self.extractor_kind_hint.setStyleSheet("font-size: 11px;")
            self._form_widget(self.extractor_editor_form, "规则名称", self.extractor_label)
            self._form_widget(self.extractor_editor_form, "扩展名", self.extractor_extensions)
            self._form_widget(self.extractor_editor_form, "索引方式", self.extractor_kind)
            self._form_widget(self.extractor_editor_form, "LLM 供应商", self.extractor_provider)
            self._form_widget(self.extractor_editor_form, "传入方式", self.extractor_input_mode)
            self._form_widget(self.extractor_editor_form, "处理 Prompt", self.extractor_prompt)
            self._form_widget(self.extractor_editor_form, "插件文件夹", self.extractor_plugin)
            self._form_widget(self.extractor_editor_form, "调用函数", self.extractor_function)
            self.extractor_editor_form.addRow(self.extractor_kind_hint)
            self.extractor_editor.setEnabled(False)
            box_layout.addWidget(self.extractor_editor)

            add.clicked.connect(self.add_extractor_rule)
            remove.clicked.connect(self.remove_extractor_rule)
            open_directory.clicked.connect(self.open_extractor_directory)
            refresh_scripts.clicked.connect(self.refresh_extractor_plugins)
            self.extractor_tree.currentItemChanged.connect(self.load_extractor_editor)
            self.extractor_label.textChanged.connect(self.apply_extractor_editor)
            self.extractor_extensions.textChanged.connect(self.apply_extractor_editor)
            self.extractor_kind.currentIndexChanged.connect(self.apply_extractor_editor)
            self.extractor_provider.currentIndexChanged.connect(self.apply_extractor_editor)
            self.extractor_input_mode.currentIndexChanged.connect(self.apply_extractor_editor)
            self.extractor_prompt.textChanged.connect(self.apply_extractor_editor)
            self.extractor_plugin.currentTextChanged.connect(self.apply_extractor_editor)
            self.extractor_function.textChanged.connect(self.apply_extractor_editor)
            self._extractor_editor_loading = False
            self._extractor_editor_item: QTreeWidgetItem | None = None
            layout.addWidget(box)
            layout.addStretch()
            return page

        def add_extractor_rule(self) -> None:
            existing_ids = {
                str((self.extractor_tree.topLevelItem(index).data(0, Qt.ItemDataRole.UserRole) or {}).get("id", ""))
                for index in range(self.extractor_tree.topLevelItemCount())
            }
            index = 1
            while f"custom-{index}" in existing_ids:
                index += 1
            item = self.add_extractor_item({
                "id": f"custom-{index}",
                "label": f"自定义规则 {index}",
                "extensions": [".custom"],
                "kind": "text",
                "enabled": True,
                "provider": "",
                "input_mode": "text",
                "prompt": "",
                "plugin": "",
                "function": "extract",
                "builtin": False,
            })
            self.extractor_tree.setCurrentItem(item)

        def remove_extractor_rule(self) -> None:
            item = self.extractor_tree.currentItem()
            if item is None:
                return
            metadata = item.data(0, Qt.ItemDataRole.UserRole) or {}
            if metadata.get("builtin"):
                QMessageBox.information(self, "内置规则", "内置规则可更改扩展名和索引方式，但不能删除")
                return
            self.extractor_tree.takeTopLevelItem(self.extractor_tree.indexOfTopLevelItem(item))

        @staticmethod
        def _extractor_kind_label(kind: str) -> str:
            return {
                "text": "直接索引文本",
                "llm": "传入 LLM",
                "python": "Python 外置插件",
            }.get(kind, kind)

        def _extractor_provider_label(self, provider_id: str) -> str:
            index = self.extractor_provider.findData(provider_id)
            return self.extractor_provider.itemText(index) if index >= 0 else (provider_id or "未选择供应商")

        def _extractor_settings_summary(self, metadata: dict[str, Any]) -> str:
            kind = str(metadata.get("kind", "python"))
            if kind == "text":
                return "确定性解析为一级正文"
            if kind == "llm":
                provider = self._extractor_provider_label(str(metadata.get("provider", "")))
                mode = "图片" if metadata.get("input_mode") == "image" else "文本"
                return f"{provider} · {mode}"
            plugin = str(metadata.get("plugin", "")).strip() or "未设置插件"
            function = str(metadata.get("function", "extract")).strip() or "extract"
            return f"{plugin} / {function}"

        def add_extractor_item(self, rule: dict[str, Any]) -> QTreeWidgetItem:
            rule_id = str(rule.get("id", ""))
            builtin = bool(rule.get("builtin", False))
            if not builtin:
                builtin = rule_id in {
                    "text", "pdf", "docx", "xlsx", "pptx", "legacy_office", "image", "zip", "eml", "mbox", "asr"
                }
            kind = str(rule.get("kind", "text")).lower()
            if kind == "builtin":
                kind = "text"
            if kind not in {"text", "llm", "python"}:
                kind = "text"
            metadata = {
                "id": rule_id,
                "builtin": builtin,
                "kind": kind,
                "provider": str(rule.get("provider", "")),
                "input_mode": str(rule.get("input_mode", "text")),
                "prompt": str(rule.get("prompt", "")),
                "plugin": str(rule.get("plugin", rule.get("script", ""))),
                "function": str(rule.get("function", "extract") or "extract"),
            }
            item = QTreeWidgetItem([
                "",
                str(rule.get("label", rule_id)),
                ", ".join(str(ext) for ext in rule.get("extensions", [])),
                self._extractor_kind_label(kind),
                self._extractor_settings_summary(metadata),
            ])
            item.setCheckState(
                0,
                Qt.CheckState.Checked if rule.get("enabled", True) else Qt.CheckState.Unchecked,
            )
            item.setData(0, Qt.ItemDataRole.UserRole, metadata)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            self.extractor_tree.addTopLevelItem(item)
            return item

        def _sync_extractor_editor_fields(self, metadata: dict[str, Any]) -> None:
            kind = str(metadata.get("kind", "python"))
            builtin = bool(metadata.get("builtin", False))
            self._set_form_field_visible(
                self.extractor_editor_form, self.extractor_provider, kind == "llm"
            )
            self._set_form_field_visible(
                self.extractor_editor_form, self.extractor_input_mode, kind == "llm"
            )
            self._set_form_field_visible(
                self.extractor_editor_form, self.extractor_prompt, kind == "llm"
            )
            self._set_form_field_visible(
                self.extractor_editor_form, self.extractor_plugin, kind == "python"
            )
            self._set_form_field_visible(
                self.extractor_editor_form, self.extractor_function, kind == "python"
            )
            image_allowed = _normalize_extractor_input_mode(
                self.extractor_extensions.text(), "image"
            ) == "image"
            image_index = self.extractor_input_mode.findData("image")
            image_item = self.extractor_input_mode.model().item(image_index)
            if image_item is not None:
                image_item.setEnabled(image_allowed)
            if kind == "text":
                hint = "直接生成一级正文；PDF、Office、压缩包和邮件仍会使用对应的确定性格式解析器。"
            elif kind == "llm":
                hint = (
                    "文本模式会先用确定性解析器或插件得到文本；图片模式把图片作为多模态输入。"
                    "原始图片仅支持 PNG、JPEG、WebP、GIF 和 BMP；PDF 或文档请使用文本模式或 Python 插件。"
                    "处理 Prompt 只对这一条扩展名规则生效。"
                )
            else:
                hint = "从插件目录加载 <文件夹>/plugin.py，并调用指定函数返回一级索引正文。"
            if builtin:
                hint += " 这是内置规则，名称与 ID 固定，但扩展名、开关和索引方式可调整。"
            self.extractor_kind_hint.setText(hint)

        def load_extractor_editor(
            self,
            item: QTreeWidgetItem | None,
            _previous: QTreeWidgetItem | None = None,
        ) -> None:
            self._extractor_editor_loading = True
            self._extractor_editor_item = item
            if item is None:
                self.extractor_editor.setEnabled(False)
                self._extractor_editor_loading = False
                return
            metadata = dict(item.data(0, Qt.ItemDataRole.UserRole) or {})
            builtin = bool(metadata.get("builtin", False))
            kind = str(metadata.get("kind", "text")).lower()
            if kind == "builtin":
                kind = "text"
            if kind not in {"text", "llm", "python"}:
                kind = "text"
            metadata["kind"] = kind
            metadata["provider"] = str(metadata.get("provider", ""))
            metadata["input_mode"] = (
                "image" if str(metadata.get("input_mode", "text")) == "image" else "text"
            )
            metadata["function"] = str(metadata.get("function", "extract") or "extract")
            self.extractor_editor.setEnabled(True)
            self.extractor_label.setReadOnly(builtin)
            self.extractor_label.setText(item.text(1))
            self.extractor_extensions.setText(item.text(2))
            self._set_combo(self.extractor_kind, kind)
            self._set_combo(self.extractor_provider, metadata["provider"])
            self._set_combo(self.extractor_input_mode, metadata["input_mode"])
            self.extractor_prompt.setPlainText(str(metadata.get("prompt", "")))
            self._set_combo(self.extractor_plugin, str(metadata.get("plugin", "")))
            self.extractor_function.setText(metadata["function"])
            self._sync_extractor_editor_fields(metadata)
            self._extractor_editor_loading = False

        def apply_extractor_editor(self, *_args: Any) -> None:
            if self._extractor_editor_loading or self._extractor_editor_item is None:
                return
            item = self._extractor_editor_item
            metadata = dict(item.data(0, Qt.ItemDataRole.UserRole) or {})
            builtin = bool(metadata.get("builtin", False))
            kind = self._combo_value(self.extractor_kind).lower()
            if kind not in {"text", "llm", "python"}:
                kind = "text"
                self._extractor_editor_loading = True
                self._set_combo(self.extractor_kind, kind)
                self._extractor_editor_loading = False
            input_mode = _normalize_extractor_input_mode(
                self.extractor_extensions.text(),
                self._combo_value(self.extractor_input_mode),
            )
            if input_mode != self._combo_value(self.extractor_input_mode):
                signals_were_blocked = self.extractor_input_mode.blockSignals(True)
                self._set_combo(self.extractor_input_mode, input_mode)
                self.extractor_input_mode.blockSignals(signals_were_blocked)
            metadata.update({
                "kind": kind,
                "provider": self._combo_value(self.extractor_provider),
                "input_mode": input_mode,
                "prompt": self.extractor_prompt.toPlainText().strip(),
                "plugin": self.extractor_plugin.currentText().strip(),
                "function": self.extractor_function.text().strip() or "extract",
            })
            if not builtin:
                item.setText(1, self.extractor_label.text().strip() or str(metadata.get("id", "")))
            item.setText(2, self.extractor_extensions.text().strip())
            item.setText(3, self._extractor_kind_label(kind))
            item.setText(4, self._extractor_settings_summary(metadata))
            item.setData(0, Qt.ItemDataRole.UserRole, metadata)
            self._sync_extractor_editor_fields(metadata)

        def open_extractor_directory(self) -> None:
            directory_text = self.extractor_dir.text().strip()
            if not directory_text:
                QMessageBox.warning(self, "无法打开目录", "Python 插件目录尚不可用")
                return
            directory = Path(directory_text).expanduser()
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                QMessageBox.warning(self, "无法打开目录", f"无法创建 Python 插件目录：{exc}")
                return
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

        def refresh_extractor_plugins(self) -> None:
            from .extractors.script import discover_python_plugins

            directory_text = self.extractor_dir.text().strip()
            current = self.extractor_plugin.currentText().strip()
            plugins: list[dict[str, Any]] = []
            if directory_text:
                directory = Path(directory_text).expanduser()
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    plugins = [
                        plugin.as_dict()
                        for plugin in discover_python_plugins(directory)
                    ]
                except OSError as exc:
                    QMessageBox.warning(self, "无法读取插件目录", str(exc))
                    return
            self._render_extractor_plugins(plugins, current=current)

        def _render_extractor_plugins(
            self,
            plugins: list[dict[str, Any]],
            *,
            current: str = "",
        ) -> None:
            self._plugin_catalog = plugins
            self.extractor_plugin.blockSignals(True)
            self.extractor_plugin.clear()
            self.extractor_plugin.addItem("输入插件文件夹名", "")
            for plugin in plugins:
                plugin_id = str(plugin.get("id", plugin.get("folder", ""))).strip()
                if not plugin_id:
                    continue
                name = str(plugin.get("name", plugin_id)).strip() or plugin_id
                available = plugin.get("available", True) is not False
                error = str(plugin.get("error", "")).strip()
                label = f"{name} ({plugin_id})"
                if not available:
                    label += f" - 不可用：{error or '插件无效'}"
                self.extractor_plugin.addItem(label, plugin_id)
                index = self.extractor_plugin.count() - 1
                detail = "\n".join(
                    value for value in (
                        str(plugin.get("description", "")).strip(),
                        str(plugin.get("path", "")).strip(),
                        error,
                    ) if value
                )
                if detail:
                    self.extractor_plugin.setItemData(
                        index, detail, Qt.ItemDataRole.ToolTipRole
                    )
            known = {
                str(plugin.get("id", plugin.get("folder", ""))).strip()
                for plugin in plugins
            }
            if current and current not in known:
                self.extractor_plugin.addItem(
                    f"{current}（当前配置，尚未发现）", current
                )
            self._set_combo(self.extractor_plugin, current)
            self.extractor_plugin.blockSignals(False)

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
            self._set_form_field_visible(self.asr_form, self.asr_device, local)
            self._set_form_field_visible(self.asr_form, self.asr_compute_type, local)
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
            current_provider_model = self._combo_value(self.provider_local_model)
            for name, fields in self.model_fields.items():
                self._populate_local_model_combo(
                    fields["local_model"],
                    current_values[name],
                    self.MODEL_CAPABILITIES[name],
                )
            self._populate_local_model_combo(self.asr_local_model, current_values["asr"], "asr")
            self._populate_local_model_combo(
                self.provider_local_model, current_provider_model, "chat"
            )

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

            intro = QLabel(
                "OCR 和语音识别现在通过插件目录中的 ocr/plugin.py 与 asr/plugin.py 执行。"
                "这里仅配置这两个随附插件依赖的服务参数；删除或替换插件文件夹后，自定义插件可完全接管处理。"
            )
            intro.setWordWrap(True)
            intro.setStyleSheet("color: palette(mid);")
            layout.addWidget(intro)

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
            self.asr_device = QComboBox()
            self.asr_device.addItem("自动", "auto")
            self.asr_device.addItem("CPU", "cpu")
            self.asr_device.addItem("CUDA", "cuda")
            self.asr_compute_type = QComboBox()
            self.asr_compute_type.addItem("int8", "int8")
            self.asr_compute_type.addItem("float16", "float16")
            self.asr_compute_type.addItem("float32", "float32")
            self.asr_compute_type.addItem("int8_float16", "int8_float16")
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
                ("本地后端", self.asr_local_backend), ("运行设备", self.asr_device),
                ("计算精度", self.asr_compute_type), ("API 模型", self.asr_model),
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
            box = QGroupBox("基于一级正文索引建立")
            form = QFormLayout(box)
            self.entities_enabled = QCheckBox("启用实体关系")
            self.entities_max_chars = self._int_box(500)
            self.entities_max_per_file = self._int_box(1)
            self.agent_max_steps = self._int_box(1)
            self.agent_max_results = self._int_box(1)
            self.agent_enabled = QCheckBox("启用 LLM 工具搜索")
            self.rag_enabled = QCheckBox("启用 RAG 语义检索")
            self.rag_max_context_chunks = self._int_box(1)
            for label, field in [
                ("", self.rag_enabled), ("RAG 最大上下文片段数", self.rag_max_context_chunks),
                ("", self.agent_enabled),
                ("", self.entities_enabled), ("实体最大文本长度", self.entities_max_chars),
                ("每文件最多实体", self.entities_max_per_file), ("Agent 最大步骤", self.agent_max_steps),
                ("Agent 最大结果数", self.agent_max_results),
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

            self._provider_editor_loading = True
            self._provider_editor_item = None
            self.provider_tree.clear()
            providers = settings.get("llm_providers", [])
            if not isinstance(providers, list):
                providers = []
            for provider in providers:
                if isinstance(provider, dict):
                    self.add_llm_provider_item(provider)
            self._provider_editor_loading = False
            self.refresh_extractor_provider_choices()
            first_provider = self.provider_tree.topLevelItem(0)
            if first_provider is not None:
                self.provider_tree.setCurrentItem(first_provider)
                self.load_llm_provider_editor(first_provider)
            else:
                self.load_llm_provider_editor(None)

            extractors = settings.get("extractors", {})
            if not isinstance(extractors, dict):
                extractors = {}
            self.extractor_dir.setText(str(
                extractors.get("plugin_dir", extractors.get("custom_dir", ""))
            ))
            self._extractor_editor_loading = True
            self._extractor_editor_item = None
            self.extractor_tree.clear()
            raw_rules = extractors.get("rules", [])
            if not isinstance(raw_rules, list):
                raw_rules = []
            for rule in raw_rules:
                if not isinstance(rule, dict):
                    continue
                self.add_extractor_item(rule)
            raw_plugins = extractors.get("plugins", [])
            plugins = [
                plugin for plugin in raw_plugins
                if isinstance(plugin, dict)
            ] if isinstance(raw_plugins, list) else []
            if plugins:
                self._render_extractor_plugins(plugins)
            else:
                self.refresh_extractor_plugins()
            self._extractor_editor_loading = False
            first_rule = self.extractor_tree.topLevelItem(0)
            if first_rule is not None:
                self.extractor_tree.setCurrentItem(first_rule)
                self.load_extractor_editor(first_rule)
            else:
                self.load_extractor_editor(None)
            model_settings = settings.get("models", {})
            if not isinstance(model_settings, dict):
                model_settings = {}
            for name, fields in self.model_fields.items():
                model = model_settings.get(name, {})
                if not isinstance(model, dict):
                    model = {}
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
            self._set_combo(self.asr_device, str(asr.get("device", "auto")))
            self._set_combo(self.asr_compute_type, str(asr.get("compute_type", "int8")))
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
            rag = settings.get("rag", {})
            self.rag_enabled.setChecked(bool(rag.get("enabled", True)))
            self.rag_max_context_chunks.setValue(int(rag.get("max_context_chunks", 8)))
            self.agent_enabled.setChecked(bool(agent.get("enabled", True)))
            self.entities_enabled.setChecked(bool(entities.get("enabled", False)))
            self.entities_max_chars.setValue(int(entities.get("max_chars", 12000)))
            self.entities_max_per_file.setValue(int(entities.get("max_per_file", 32)))
            self.agent_max_steps.setValue(int(agent.get("max_steps", 6)))
            self.agent_max_results.setValue(int(agent.get("max_results", 12)))

        def payload(self) -> dict[str, Any]:
            if self.chunk_overlap.value() >= self.chunk_size.value():
                raise ValueError("向量分块重叠必须小于分块大小")
            folders = [self.folder_list.item(index).text() for index in range(self.folder_list.count())]
            extractor_rules: list[dict[str, Any]] = []
            for row_index in range(self.extractor_tree.topLevelItemCount()):
                item = self.extractor_tree.topLevelItem(row_index)
                metadata = item.data(0, Qt.ItemDataRole.UserRole) or {}
                rule = {
                    "id": str(metadata.get("id", "")),
                    "label": item.text(1),
                    "kind": str(metadata.get("kind", "text")),
                    "enabled": item.checkState(0) == Qt.CheckState.Checked,
                    "extensions": [value.strip() for value in item.text(2).split(",") if value.strip()],
                    "provider": str(metadata.get("provider", "")),
                    "input_mode": str(metadata.get("input_mode", "text")),
                    "prompt": str(metadata.get("prompt", "")),
                    "plugin": str(metadata.get("plugin", "")),
                    "function": str(metadata.get("function", "extract") or "extract"),
                }
                extractor_rules.append(rule)
            llm_providers: list[dict[str, Any]] = []
            for row_index in range(self.provider_tree.topLevelItemCount()):
                item = self.provider_tree.topLevelItem(row_index)
                metadata = item.data(0, Qt.ItemDataRole.UserRole) or {}
                provider = {
                    "id": str(metadata.get("id", "")),
                    "name": item.text(1),
                    "enabled": item.checkState(0) == Qt.CheckState.Checked,
                    "mode": str(metadata.get("mode", "openai")),
                    "local_model": str(metadata.get("local_model", "")),
                    "base_url": str(metadata.get("base_url", "")),
                    "model": str(metadata.get("model", "")),
                }
                api_key = str(metadata.get("api_key", "")).strip()
                if api_key:
                    provider["api_key"] = api_key
                if metadata.get("clear_api_key"):
                    provider["clear_api_key"] = True
                llm_providers.append(provider)
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
                "device": self._combo_value(self.asr_device),
                "compute_type": self._combo_value(self.asr_compute_type),
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
                "extractors": {"rules": extractor_rules},
                "llm_providers": llm_providers,
                "models": models,
                "ocr": ocr,
                "asr": asr,
                "entities": {
                    "enabled": self.entities_enabled.isChecked(),
                    "max_chars": self.entities_max_chars.value(),
                    "max_per_file": self.entities_max_per_file.value(),
                },
                "agent": {
                    "enabled": self.agent_enabled.isChecked(),
                    "max_steps": self.agent_max_steps.value(),
                    "max_results": self.agent_max_results.value(),
                },
                "rag": {
                    "enabled": self.rag_enabled.isChecked(),
                    "max_context_chunks": self.rag_max_context_chunks.value(),
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
            status_action = QAction("索引状态", self)
            status_action.triggered.connect(self.show_status_panel)
            reindex_action = QAction("重新索引", self)
            reindex_action.triggered.connect(lambda: self.start_index(False))
            rebuild_action = QAction("完整重建", self)
            rebuild_action.triggered.connect(lambda: self.start_index(True))
            toolbar = self.addToolBar("操作")
            toolbar.setMovable(False)
            view_menu = self.menuBar().addMenu("查看")
            view_menu.addAction(status_action)
            view_menu.addAction(settings_action)
            toolbar.addAction(status_action)
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
            progress_row = QHBoxLayout()
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setTextVisible(True)
            self.progress_detail = QLabel()
            self.progress_detail.setMinimumWidth(230)
            self.progress_detail.setStyleSheet("color: palette(mid); font-size: 11px;")
            progress_row.addWidget(self.progress_bar, 1)
            progress_row.addWidget(self.progress_detail)
            layout.addLayout(progress_row)

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
            if not status["models"].get("rag", True):
                semantic = "关"
            pieces.append(f"语义搜索 {semantic}")
            pieces.append(f"一级 LLM {status['models'].get('llm_providers', 0)} 个")
            pieces.append(f"Python 插件 {status.get('capabilities', {}).get('plugins', 0)} 个")
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
            progress = status.get("progress") or {}
            self.progress_bar.setValue(int(progress.get("percent", 0)))
            phase_labels = {
                "preparing": "准备完整重建", "scanning": "扫描目录", "indexing": "提取与索引",
                "embedding": "生成向量", "entities": "抽取实体", "complete": "本轮索引完成",
                "failed": "本轮索引失败", "idle": "空闲",
            }
            self.progress_bar.setFormat(f"{phase_labels.get(progress.get('phase'), '索引状态')} %p%")
            self.progress_detail.setText(str(progress.get("current_file") or ""))

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

        def show_status_panel(self) -> None:
            dialog = QDialog(self)
            dialog.setWindowTitle("Semdex 索引状态")
            dialog.resize(760, 520)
            layout = QVBoxLayout(dialog)
            intro = QLabel("实时查看当前阶段、文件队列、已启用能力和最近一次索引结果。")
            intro.setWordWrap(True)
            intro.setStyleSheet("color: palette(mid);")
            layout.addWidget(intro)
            progress = QProgressBar()
            progress.setRange(0, 100)
            layout.addWidget(progress)
            phase = QLabel()
            phase.setWordWrap(True)
            phase.setStyleSheet("color: palette(mid);")
            layout.addWidget(phase)
            details = QTreeWidget()
            details.setColumnCount(2)
            details.setHeaderLabels(["项目", "当前状态"])
            details.setRootIsDecorated(True)
            details.setAlternatingRowColors(True)
            details.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            details.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            layout.addWidget(details, 1)
            close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            close.rejected.connect(dialog.reject)
            layout.addWidget(close)

            def refresh_dialog() -> None:
                try:
                    status = self.controller.status()
                    snapshot = status.get("progress") or {}
                    progress.setValue(int(snapshot.get("percent", 0)))
                    phase_labels = {
                        "preparing": "准备完整重建", "scanning": "扫描目录",
                        "indexing": "提取与一级索引", "embedding": "生成语义向量",
                        "entities": "抽取实体关系", "complete": "本轮索引完成",
                        "failed": "本轮索引失败", "idle": "空闲",
                    }
                    current_file = str(snapshot.get("current_file") or "")
                    phase.setText(
                        f"{phase_labels.get(snapshot.get('phase'), '索引状态')}"
                        + (f"\n当前文件：{current_file}" if current_file else "")
                    )

                    def add_section(title: str, entries: list[tuple[str, object]]) -> None:
                        root = QTreeWidgetItem([title, ""])
                        root.setFirstColumnSpanned(True)
                        root.setExpanded(True)
                        for label, value in entries:
                            root.addChild(QTreeWidgetItem([label, str(value)]))
                        details.addTopLevelItem(root)

                    details.clear()
                    files = status.get("files") or {}
                    by_status = files.get("by_status") or {}
                    add_section("文件状态", [
                        ("文件总数", files.get("total", 0)),
                        ("已完成一级索引", by_status.get("done", 0)),
                        ("待索引", by_status.get("pending", 0)),
                        ("等待 LLM", by_status.get("waiting_model", 0)),
                        ("等待本地能力", by_status.get("waiting_capability", 0)),
                        ("失败", by_status.get("failed", 0)),
                        ("跳过", by_status.get("skipped", 0)),
                    ])
                    models = status.get("models") or {}
                    capabilities = status.get("capabilities") or {}
                    add_section("能力状态", [
                        ("一级 LLM 供应商", f"{models.get('llm_providers', 0)} 个启用"),
                        ("RAG 语义检索", "已启用" if models.get("rag") else "未启用"),
                        ("LLM 工具搜索", "已启用" if models.get("agent") else "未启用"),
                        ("实体关系", "已启用" if models.get("entities") else "未启用"),
                        ("Python 插件", f"{capabilities.get('plugins', 0)} 个启用"),
                    ])
                    folders = status.get("folders") or []
                    add_section("索引目录", [("目录", folder) for folder in folders] or [("目录", "尚未设置")])
                    last_run = status.get("last_run") or {}
                    if last_run.get("error"):
                        recent = [("状态", "失败"), ("错误", last_run["error"])]
                    elif not last_run:
                        recent = [("状态", "暂无运行记录")]
                    else:
                        scan = last_run.get("scan") or {}
                        indexed = last_run.get("index") or {}
                        recent = [
                            ("模式", "完整重建" if last_run.get("full_rebuild") else "增量索引"),
                            ("扫描到文件", scan.get("scanned", 0)),
                            ("新增或变更", scan.get("new_or_changed", 0)),
                            ("一级正文完成", indexed.get("indexed", 0)),
                            ("向量化文件", indexed.get("embedded_files", 0)),
                            ("实体抽取文件", indexed.get("entities_indexed", 0)),
                        ]
                        if last_run.get("embedding_error"):
                            recent.append(("向量错误", last_run["embedding_error"]))
                    add_section("最近一次运行", recent)
                except Exception as exc:
                    details.clear()
                    details.addTopLevelItem(QTreeWidgetItem(["无法读取状态", str(exc)]))

            timer = QTimer(dialog)
            timer.timeout.connect(refresh_dialog)
            timer.start(800)
            refresh_dialog()
            dialog.exec()

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
