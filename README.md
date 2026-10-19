# ComfyUI Outfit Reference Composer

用于将单品图合成为服饰穿搭参考图的 ComfyUI 节点。

节点把完整、正面的服饰单品图排列到一套普通成人的“无人物坐标系”中，输出一张无人穿搭参考图，供 Flux 等生图模型与全身正面模特图共同使用。

仓库同时提供 V1 语义锚点版、V2 固定分区版、V3 标签重叠版和 V3_TOP 上半身专用版。四个节点使用独立的节点 ID 和输入结构，可以安装在同一个 ComfyUI 中，并在同一工作流内同时使用。

## V1（语义锚点版）主要特性

- 一个节点接收上装、下装、袜子、鞋、帽子、包、眼镜、项链、耳环和手镯。
- 上装使用颈口、肩宽与下摆语义；下装使用腰线、版型与下摆语义。
- 所有单品只做等比缩放，不会为了命中安全框而横向或纵向拉伸。
- 上装、下装、袜子和鞋按语义间距紧凑排列，实际安全框互不重叠，单品始终完整可见。
- 项链最后绘制，是唯一允许覆盖上装的类别。
- 包固定在画面右侧，并以提手顶部对齐手部高度；手镯默认放在画面左侧、与手腕同高的独立区域。
- 帽子、鞋和手腕饰品按约五头身的 3D 卡通人物比例校准；手串与手镯按可见环体高度归一，避免源图留白造成夸张尺寸差。
- 鞋和袜子直接使用输入图中的完整一对，不复制、不镜像单只。
- `show_layout_guides` 可以显示安全区、真实输出框和关键锚点；正式出图时关闭即可。
- 自动过滤白色背景中的离散水印和噪点，避免它们撑大前景范围。

## V3_TOP（Upper Body Only，无 JSON）

节点名称为 `Outfit Reference Composer V3_TOP (Upper Body Only)`。

- 只提供 `top`、`hat`、`glasses`、`earrings` 和 `necklace` 五个图片输入，不提供 `bottom`、`socks`、`shoes`、`bag` 或 `bracelet` 输入。
- 输入端口名是唯一类别信息，不接收也不解析 `outfit_spec`。
- 帽子、眼镜／耳环、颈饰／上装按语义行排版；缺少某一行时，相邻语义行会使用释放出的纵向空间。例如未选择眼镜和耳环时，帽子会在原帽子＋面部区域内下移并放大。
- 布局专门面向上装以上区域，五类单品在保持完整、等比缩放的前提下尽量占满画布；帽子、眼镜、颈饰和上装的主列统一居中。
- `necklace` 最后绘制，可以覆盖 `top` 的领口和上胸区域；其他类别保持各自固定位置。
- 白底清理、透明通道读取、`contain` 等比适配、`background_threshold`、`slot_padding` 和 `show_layout_guides` 的逻辑与 V2 相同。
- 五类都连接时仍使用原 V3_TOP 坐标与请求画布尺寸；有空槽时，正式输出会按真实 alpha 占用范围保留安全边距，并向外对齐到 64 像素倍数后裁掉外围空白。开启 `show_layout_guides` 时保留完整请求画布，便于检查坐标。
- V3_TOP 是新增的独立节点，不会替换或修改 V2；V2 仍保留原有十个类别输入及原有排版逻辑。

## V3（Tagged Overlap，全身版）

节点名称为 `Outfit Reference Composer V3 (Tagged Overlap)`。

- 保留 V2 的十个图片输入、清底、等比缩放、固定附件分区和调试框逻辑。
- 额外接收 Wardrobe Library 输出的 `layout_tags_json`；用户仍只需在衣橱库里选择单品，不需要手填长度数值。
- `top = top_coverwaist` 时，以最终下装的实际紧裁切框为基准：上装保持原上沿、沿中心轴等比放大，下摆精确到下装高度 30% 的胯骨线，并在下装之后绘制。
- `bottom = bottom_shoehalf` 时，以鞋的实际紧裁切框为基准：裤装保持原上沿与腰部中心轴，并按真实可见 alpha 下沿等比延伸到鞋高 2/3 的位置（透明留边不参与落点计算），再向鞋层覆盖一小段抗锯齿安全量；鞋图本身仍严格只保留下方 1/3，避免宽鞋或靴子从裤腿两侧露出鞋帮，同时消除透明留边与弧形轮廓造成的画布白缝。
- 两种标签可以同时生效；某个标签缺少必要的相邻图片时，只忽略该遮挡标签，仍执行 V3 的空槽动态布局。
- 帽子、眼镜／耳环、上装／颈饰、下装、袜子／鞋各自属于语义行。缺少某行时，仅相邻语义行使用释放出的纵向空间；包和手饰仍留在各自侧栏，不做无语义的跨区搬移。
- 十类都连接且没有遮挡标签时，V3 仍直接走 V2，保持原 placement 与像素输出。存在空槽时，正式输出按真实 alpha 占用范围加安全边距并向外对齐到 64 像素倍数，裁掉外围空白；`show_layout_guides` 会保留完整请求画布。
- V3、V2 与 V3_TOP 是三个独立节点，不会互相覆盖。

## V2（Fixed Slots，无 JSON）

节点名称为 `Outfit Reference Composer V2 (Fixed Slots)`。

- 保留 `top`、`bottom`、`socks`、`shoes`、`hat`、`bag`、`glasses`、`necklace`、`earrings` 和 `bracelet` 十个图片输入。
- 输入端口名是唯一类别信息，不接收也不解析 `outfit_spec`。
- 每个类别使用永久固定分区；项链／颈饰有意覆盖上装的领口与上胸区域并最后绘制，其余分区互不重叠。未连接的类别保持空白，不会引起其他单品移动。
- 单品先去除白底或读取透明通道，再裁掉源图留白，以 `contain` 方式等比最大化放进对应分区。
- 不根据高腰／低腰、宽松／修身、长裤／短裤等描述改变比例，也不添加合成阴影。
- `background_threshold` 统一控制白底识别阈值，`slot_padding` 控制每个固定框的内边距。
- `show_layout_guides` 开启时，红框表示固定分区，蓝框表示单品实际输出范围。
- 袜子固定在鞋子左侧，与鞋子共享画面底部纵向区域；鞋框因此拥有更大的高度，短鞋与长靴都会完整显示。
- 颈饰、上装、下装和鞋子共用画布宽度 46% 处的纵向中心轴；调试图中的灰色虚线可用于核对，正式出图不会显示。袜子相应左移以保持与鞋子不重叠。
- 下装按图片上部的裤腰／裙腰中心落在共用轴上，而非用可能包含侧链、非对称裙摆或地面阴影的整张外框居中；有框图以绿色十字标出腰线锚点。对齐时仍只做等比缩放，且保留完整单品在安全框内。

若业务需要 Basic 默认服饰，应由上游把相应 Basic 图片接入缺少的 `top`、`bottom` 或 `shoes` 端口；V2 不负责按 `item_id` 选择图片。

## 安装

将本仓库目录放到 `ComfyUI/custom_nodes/`，然后重启 ComfyUI。

节点只使用 ComfyUI 已包含的 `torch`、`numpy` 和 `Pillow`，不需要安装额外依赖。

升级旧版本后如果节点上仍看不到 `socks` 输入，请先重启 ComfyUI、强制刷新浏览器页面，再删除旧的节点实例并重新添加。ComfyUI 工作流会保存旧节点的输入结构，仅刷新已有实例不一定更新接口。

## 在 ComfyUI 中测试

### V3 Tagged Overlap

1. 添加 `Wardrobe Library Selector / 可视化衣橱库` 和 `Outfit Reference Composer V3 (Tagged Overlap)`。
2. 将衣橱库的图片输出接到 V3 同名图片输入。
3. 将衣橱库最后的 `layout_tags_json` 输出接到 V3 的同名输入；不要把 `selected_items_json` 接错。
4. 建议先使用默认 `1536 × 2048`、`background_threshold = 24`、`slot_padding = 10`，并开启 `show_layout_guides` 检查裤头和鞋面两处重叠。调试模式固定保留完整画布。
5. 确认效果后关闭调试框；有空槽的正式结果会自动裁掉外围空白并保持 64 像素倍数，再将 `outfit_reference` 接到下游图片参考输入。

### V3_TOP Upper Body Only

1. 添加 `Outfit Reference Composer V3_TOP (Upper Body Only)`；它可以与 V2 节点同时放在同一工作流中。
2. 仅为需要的 `top`、`hat`、`glasses`、`earrings` 和 `necklace` 创建 `Load Image`，并接到节点同名输入。
3. 不需要 JSON；建议先使用默认 `1536 × 2048`、`background_threshold = 24` 和 `slot_padding = 10`。
4. 初次测试开启 `show_layout_guides`，确认主列居中、动态语义行和颈饰覆盖关系；确认后关闭。关闭后有空槽的结果会自动做内容裁切。
5. 将唯一输出 `outfit_reference` 接到 Flux 的图片参考输入。

### V2 Fixed Slots

1. 添加 `Outfit Reference Composer V2 (Fixed Slots)`。
2. 为所选单品创建 `Load Image`，接到节点同名输入。
3. 不需要 JSON；建议先使用默认 `1536 × 2048`、`background_threshold = 24` 和 `slot_padding = 10`。
4. 初次测试开启 `show_layout_guides`；确认后关闭。
5. 将唯一输出 `outfit_reference` 接到 Flux 的图片参考输入。

### V1 语义锚点版

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

手镯不写位置时默认放在画面左侧；可用 `"side": "right"` 或 `"placement": "right_wrist"` 明确放到右侧。包保持在画面右侧手持区域。

未知的扩展字段会被忽略，方便后端逐步增加描述信息；但上表中的已知字段如果拼写错误，节点会直接报出允许值，避免静默排错。

## 锚点与安全区逻辑

- 上装将 `fit` 对应的目标肩宽与 `length` 对应的目标高度合成为一个统一缩放倍数，横纵轴始终使用同一个倍数。
- 下装从透明轮廓上部测量局部胯宽，再把 `fit + silhouette` 的目标胯宽与 `length` 的目标高度合成为一个统一缩放倍数；`high / mid / low` 腰线由同一条 `hip axis` 向上取不同偏移。
- `source_shoulder_ratio`（`0.30`～`1.0`）可覆盖特殊上装的默认肩宽比例；`source_hip_width_ratio`（`0.10`～`1.0`）可覆盖特殊下装的自动胯宽测量。
- 上装、下装、袜子和鞋先完成等比测量，再依次紧凑排版。露脐组合保留较大的腹部语义间距；短裙/短裤到鞋之间保留有限的腿部语义间距。
- 整套主列超过画布高度时，所有主列单品使用同一个额外缩放系数一起缩小，不会单独压扁任何单品。
- 主列缺少中间类别时会保留对应身体区段：例如只连接 `top + shoes`，鞋仍留在脚部区域，不会被吸到上装下方。
- 调试图中的 `shoulder` 与 `hip axis` 虚线是统一的人体参考轴；红框是排版后的有效安全区，蓝框是等比缩放后的真实单品框。
- 所有单品仍受画布与横向安全区限制，节点不会用裁切来隐藏越界部分。
- 帽子、眼镜、耳环、手镯、包和鞋使用约五头身 3D 卡通人物的参考尺度；这些坐标只表达穿戴位置，不代表画面中存在人物。

`coverage` 是对 `length` 的一致性约束：`show_midriff` 强制使用 `crop`，`cover_waist` 至少使用 `waist`，`cover_hip` 至少使用 `hip`。互相矛盾时以 `coverage` 为准。

这一版仍是确定性的启发式布局，不会识别真实人体或真实服装缝线。严格等比意味着：当某张源图本身的长宽比例与描述目标冲突时，节点优先保证“不拉伸、完整、无重叠”，肩/胯尺度其次，长度作为语义目标；不会为了同时贴满宽高锚点而破坏单品比例。旧 JSON 中的 `max_axis_distortion` 可以保留，但第二轮起不会再允许它开启非等比拉伸。

浅色单品难以从白底分离时，可以添加 `background_threshold` 调整背景阈值；有效范围 `4`～`100`，建议先尝试 `14`～`34`。

如果 `calf` / `ankle` / `floor` 长度的裤或裙与袜子同时输入，节点会报出明确错误，因为它们无法在“不遮挡”的身体坐标中同时占用腿部区域。袜子应搭配 `thigh` / `knee` 长度的短裤或短裙。

## 输入图片要求

- 单品正面、完整、居中。
- 鞋与袜子为已经摆好的完整一对。
- 每个 slot 只接一张图片；批量 IMAGE 需要先拆分。
- 白色或浅灰色背景效果最好，节点会自动做 MVP 背景清理。
- V3 / V3_TOP 接到能输出四通道 `IMAGE` 的上游节点时，会直接使用源 alpha 做紧裁切与拼合，不再按 RGB 重新抠背；三通道白底图片仍走原背景清理逻辑。
- 标准 ComfyUI `Load Image` 的具体通道行为随版本而异。需要稳定保留透明通道时，请使用明确输出 RGBA `IMAGE` 的衣橱库节点；无需另接 mask 节点。
