# ComfyUI Outfit Reference Composer

用于将单品图合成为服饰穿搭参考图的 ComfyUI 节点。

节点把完整、正面的服饰单品图排列到一套普通成人的“无人物坐标系”中，输出一张无人穿搭参考图，供 Flux 等生图模型与全身正面模特图共同使用。

## 主要特性

- 一个节点接收上装、下装、袜子、鞋、帽子、包、眼镜、项链、耳环和手镯。
- 上装使用颈口、肩宽与下摆语义；下装使用腰线、版型与下摆语义。
- 上装、下装、袜子和鞋的实际安全框互不重叠，单品始终完整可见。
- 项链最后绘制，是唯一允许覆盖上装的类别。
- 包固定在画面右侧手持区域；手镯放在手腕同高的独立区域。
- 鞋和袜子直接使用输入图中的完整一对，不复制、不镜像单只。
- `show_layout_guides` 可以显示安全区、真实输出框和关键锚点；正式出图时关闭即可。
- 自动过滤白色背景中的离散水印和噪点，避免它们撑大前景范围。

## 安装

将本仓库目录放到 `ComfyUI/custom_nodes/`，然后重启 ComfyUI。

节点只使用 ComfyUI 已包含的 `torch`、`numpy` 和 `Pillow`，不需要安装额外依赖。

## 在 ComfyUI 中测试

1. 为所选单品分别创建 `Load Image`。
2. 把图片接到节点同名输入，例如 `top`、`bottom`、`socks`、`shoes`。
3. 将对应 JSON 的完整内容粘贴到 `outfit_spec`。
4. 初次测试开启 `show_layout_guides`；确认后关闭。
5. 将唯一输出 `outfit_reference` 接到 Flux 的图片参考输入。

调试图中红框是该单品的安全框，蓝框是实际缩放后的完整单品框，灰色虚线是人体参考锚线。

业务单品图片与混搭测试 JSON 不随公开仓库发布，请使用团队内部维护的测试文档。

## JSON 结构

```json
{
  "items": {
    "top": {
      "item_id": "top1",
      "slot": "top",
      "garment_type": "jersey",
      "sleeve_length": "short",
      "fit": "loose",
      "length": "crop",
      "coverage": "show_midriff"
    },
    "bottom": {
      "item_id": "bottom0",
      "slot": "bottom",
      "garment_type": "shorts",
      "fit": "loose",
      "waist": "low",
      "length": "knee",
      "silhouette": "wide"
    },
    "socks": {
      "item_id": "shoe0",
      "slot": "socks",
      "length": "mid_calf"
    },
    "shoes": {
      "item_id": "shoe3",
      "slot": "shoes",
      "pair_mode": "source_pair"
    }
  }
}
```

JSON 必须是一个以 slot 为 key 的对象；不能在同一个对象中重复写多个 `item_id`。

## 正式词表

| 类别 | 字段 | 支持值 |
| --- | --- | --- |
| 上装 | `garment_type` | `tank`, `camisole`, `jersey`, `tee`, `shirt`, `hoodie`, `jacket`, `outerwear`, `layered_outerwear` |
| 上装 | `sleeve_length` | `sleeveless`, `cap`, `short`, `elbow`, `long` |
| 上装 | `fit` | `slim`, `regular`, `loose`, `oversized` |
| 上装 | `length` | `crop`, `waist`, `hip`, `upper_thigh` |
| 上装 | `coverage` | `show_midriff`, `cover_waist`, `cover_hip` |
| 下装 | `garment_type` | `shorts`, `pants`, `skirt` |
| 下装 | `waist` | `high`, `mid`, `low` |
| 下装 | `length` | `thigh`, `knee`, `calf`, `ankle`, `floor` (`full` 兼容为 `floor`) |
| 下装 | `fit` | `slim`, `regular`, `loose`, `oversized` |
| 下装 | `silhouette` | `slim`, `straight`, `wide`, `baggy`, `flare`, `skirt`, `pleated`, `a_line`, `voluminous` |
| 袜子 | `length` | `ankle`, `crew`, `mid_calf`, `knee`, `knee_high` |
| 配饰 | `size` | `tiny`, `small`, `medium`, `large` |

未知的扩展字段会被忽略，方便后端逐步增加描述信息；但上表中的已知字段如果拼写错误，节点会直接报出允许值，避免静默排错。

## 锚点与安全区逻辑

- `fit` 只改变目标肩宽或下装宽度，不再改变衣长。
- `length` 只选择目标下摆，不再改变横向安全区。
- `waist` 只选择下装起点，高腰、中腰、低腰不会被图片宽高比覆盖。
- 上装按颈口—下摆和目标肩宽在两个方向独立归一化；长袖与短袖通过不同的“肩宽占整图宽度比例”换算。
- 下装宽度完全由 `fit + silhouette` 决定，不再受商品图留白或原始长宽比影响。
- 下装以腰线和下摆为硬纵向锚点，图片构图不会把低腰裤重新推到高腰位置。
- 所有缩放完成后仍受硬安全区限制，节点不会用裁切来隐藏越界部分。
- 调试图中的 `shoulder` 虚线是人体肩高参考；真正参与上装缩放的是描述对应的目标肩宽。

`coverage` 是对 `length` 的一致性约束：`show_midriff` 强制使用 `crop`，`cover_waist` 至少使用 `waist`，`cover_hip` 至少使用 `hip`。互相矛盾时以 `coverage` 为准。

这一版是确定性的启发式布局，不会识别真实服装关键点：抠图顶边近似作为颈口位置，肩宽由 `garment_type` / `sleeve_length` 的预设比例估算。因此高领、连帽、立领或特殊构图仍可能需要调参。`source_shoulder_ratio`（`0.30`～`1.0`）可以覆盖默认肩宽比例。

为了同时命中横纵锚点，上下装可能进行有限的非等比归一化。若两轴所需变形超过默认 `2.5` 倍，节点会报错而不是破坏版型；确认属于特殊单品时可通过 `max_axis_distortion`（`1.0`～`4.0`）调整上限。

浅色单品难以从白底分离时，可以添加 `background_threshold` 调整背景阈值；有效范围 `4`～`100`，建议先尝试 `14`～`34`。

如果 `calf` / `ankle` / `floor` 长度的裤或裙与袜子同时输入，节点会报出明确错误，因为它们无法在“不遮挡”的身体坐标中同时占用腿部区域。袜子应搭配 `thigh` / `knee` 长度的短裤或短裙。

## 输入图片要求

- 单品正面、完整、居中。
- 鞋与袜子为已经摆好的完整一对。
- 每个 slot 只接一张图片；批量 IMAGE 需要先拆分。
- 白色或浅灰色背景效果最好，节点会自动做 MVP 背景清理。
- 标准 ComfyUI `Load Image` 的 `IMAGE` 不携带独立 alpha mask。透明 PNG 建议先铺成白底；只有明确能输出 RGBA `IMAGE` 的上游节点才会让本节点直接使用 alpha。
