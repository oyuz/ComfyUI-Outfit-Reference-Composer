# 五组 MVP 混搭测试

每个 JSON 文件都可以完整粘贴到节点的 `outfit_spec`。按表格把对应单品图接到同名输入即可。

| 文件 | ComfyUI 图片连接 | 主要覆盖场景 |
| --- | --- | --- |
| `01_crop_shorts_socks.json` | `top=top1`, `bottom=bottom0`, `socks=shoe0`, `shoes=shoe3`, `hat=acc25`, `bag=acc8`, `glasses=acc10` | 宽松露脐上衣、低腰短裤、袜子 |
| `02_tank_shorts_socks.json` | `top=top5`, `bottom=bottom5`, `socks=shoe0`, `shoes=shoe10`, `necklace=acc16`, `earrings=acc23`, `bracelet=acc22` | 修身无袖露脐上衣、高腰短裤、袜子 |
| `03_regular_mini_skirt.json` | `top=top4`, `bottom=bottom11`, `shoes=shoe18`, `bag=acc1`, `glasses=acc13`, `necklace=acc19` | 常规腰长上衣、大腿长度百褶裙 |
| `04_long_shirt_ankle_skirt.json` | `top=top10`, `bottom=bottom16`, `shoes=shoe18`, `hat=acc24`, `bag=acc3`, `glasses=acc15`, `necklace=acc17` | 宽松长袖及臀衬衫、脚踝长裙 |
| `05_oversized_floor_pants.json` | `top=top36`, `bottom=bottom27`, `shoes=shoe24`, `hat=acc26`, `bag=acc7`, `glasses=acc12`, `bracelet=acc21` | oversize 大腿上部长度外套、低腰拖地裤 |

`shoe0` 在前两组中有意连接到新增的 `socks` 输入。鞋和袜子的输入图都必须已经包含完整、正面摆放的一对；节点不会把单只镜像成一对。
