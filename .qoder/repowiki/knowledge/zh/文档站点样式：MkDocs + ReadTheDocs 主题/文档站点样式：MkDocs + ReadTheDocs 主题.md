---
kind: frontend_style
name: 文档站点样式：MkDocs + ReadTheDocs 主题
category: frontend_style
scope:
    - '**'
source_files:
    - mkdocs.yml
---

本仓库不包含任何前端 UI 代码（无 CSS/SCSS/Tailwind、无组件库、无设计令牌），唯一的“前端风格”体现在基于 MkDocs 构建的文档站点，采用 vLLM 风格的导航结构。

- 使用的系统/工具：MkDocs 静态站点生成器，主题使用内置 `readthedocs`；通过 `mkdocs.yml` 配置站点名称、导航树与 Markdown 扩展（admonition、toc、tables、fenced_code、attr_list）。
- 关键文件：`mkdocs.yml`（站点与主题配置）、`docs/` 目录下的全部 `.md` 文档（按 vLLM 风格划分为 User Guide / Getting Started / Serving / Deployment / Configuration / Models / Features / Developer Guide / Design Documents / Benchmarking / API Reference / CLI Reference / Community / Governance 等板块）。
- 架构与约定：文档以纯 Markdown 组织，不引入自定义 CSS 或第三方主题，视觉风格完全由 ReadTheDocs 默认主题决定；导航层级在 `nav` 字段中集中声明，保持与 vLLM 文档一致的目录划分方式。
- 开发者应遵循的规则：新增页面时仅追加 Markdown 文件并在 `mkdocs.yml` 的 `nav` 中注册对应条目，不要自行编写 CSS 或替换主题。