# eLLM 文档

这里是文档站点首页。当前结构已经整理成 MkDocs 可直接使用的目录形态，后续可以把现有内容逐步迁移到对应章节。

## 快速入口

- [开始使用](getting_started/index.md)
- [配置](configuration/index.md)
- [服务](serving/index.md)
- [CLI](cli/index.md)
- [功能](features/index.md)
- [示例](examples/index.md)
- [贡献](contributing/index.md)

## 迁移说明

为了贴近 vLLM 的文档组织方式，本站点采用了按主题分区的结构，而不是把所有 Markdown 平铺在 `docs/` 根目录。

## 下一步

- 把现有章节内容移动到对应子目录
- 补充每个章节的二级页面
- 继续完善 MkDocs 主题、搜索和导航
# Welcome to eLLM

This documentation set is being reorganized around MkDocs and aligned with the
section layout used by vLLM.

Use the navigation to move through the same major buckets vLLM exposes:

- Getting started
- User guide
- Serving and deployment
- Models and features
- Developer docs
- API and CLI references

The pages in this repository are intentionally being split into smaller,
MkDocs-friendly files so the site can grow without a monolithic markdown index.

## Preview locally

Use `mkdocs serve` to preview the site while the migration is in progress.

## Start here

- [Quickstart](getting_started/quickstart.md)
- [Configuration](configuration/index.md)
- [Optimization](configuration/optimization.md)
- [Model contribution basics](contributing/model/basic.md)
