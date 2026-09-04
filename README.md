# ComfyUI Outfit Reference Composer

将最多 9 张完整单品图确定性拼成一张“无人穿搭参考图”，用于与全身正面的 3D 卡通模特图一起输入 Flux 换装流程。

首版只支持 `female_front` 与 `male_front`。包固定右手手持；鞋必须输入一张已经成对、正面摆放的图。

## 安装

将本目录放到 `ComfyUI/custom_nodes/` 后重启 ComfyUI。节点仅使用 ComfyUI 自带的 `torch`、`numpy` 和 `Pillow`，无需额外安装依赖。

## 测试工作流

将 `Load Image` 分别接到节点的 `top`、`bottom`、`shoes` 等可选输入；将下面 JSON 粘贴到 `outfit_spec`：

```json
{
  "items": {
    "top": {"node_layout": {"silhouette": "jacket_with_inner", "fit": "regular", "length": "waist"}},
    "bottom": {"node_layout": {"silhouette": "wide_pants", "fit": "loose", "length": "full"}},
    "shoes": {"node_layout": {"placement": "feet", "pair_mode": "source_pair"}}
  }
}
```

输出 `outfit_reference` 接到 Flux 的图片参考输入；`outfit_mask` 与 `outfit_prompt` 为可选输出。

## 当前限制

- 单品图需为正面、完整、居中、浅色/白色背景。
- MVP 使用“接近白色即背景”的简单抠图。生产版应替换为外部背景移除节点或接收 alpha mask。
- `node_layout` 目前使用 `fit` 微调宽度；下一版会把长度、层级和饰品尺寸做成更精确的规则。
