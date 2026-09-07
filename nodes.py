import json
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageDraw

SLOTS=("top","bottom","shoes","hat","bag","glasses","necklace","earrings","bracelet")
class OutfitReferenceComposer:
 @classmethod
 def INPUT_TYPES(cls):
  return {"required":{"outfit_spec":("STRING",{"multiline":True,"default":"{\n  \"items\": {}\n}"}),"width":("INT",{"default":768,"min":256,"max":2048,"step":64}),"height":("INT",{"default":1024,"min":256,"max":2048,"step":64}),"background":(["off_white","white","light_gray"],{"default":"off_white"}),"show_layout_guides":("BOOLEAN",{"default":False})},"optional":{s:("IMAGE",) for s in SLOTS}}
 RETURN_TYPES=("IMAGE",); RETURN_NAMES=("outfit_reference",); FUNCTION="compose"; CATEGORY="Outfit Reference"
 @staticmethod
 def _image(t): return Image.fromarray(np.clip(t[0].detach().cpu().numpy()*255,0,255).astype(np.uint8)).convert("RGB")
 @staticmethod
 def _cutout(image):
  marker=image.copy(); bg=(0,0,0)
  for p in ((0,0),(image.width-1,0),(0,image.height-1),(image.width-1,image.height-1)): ImageDraw.floodfill(marker,p,bg,thresh=24)
  alpha=np.where(np.all(np.asarray(marker)==bg,axis=2),0,255).astype(np.uint8); box=Image.fromarray(alpha).getbbox()
  if not box: return None
  out=image.convert("RGBA"); out.putalpha(Image.fromarray(alpha)); return out.crop(box)
 @staticmethod
 def _parse(spec):
  try: value=json.loads(spec)
  except json.JSONDecodeError as e: raise ValueError(f"outfit_spec is not valid JSON: {e.msg}") from e
  items=value.get("items",value)
  if not isinstance(items,dict): raise ValueError("outfit_spec.items must be an object keyed by slot")
  return items
 @staticmethod
 def _zone(slot,length,coverage=None):
  if slot=="top":
   if coverage=="show_midriff": return (.27,.17,.73,.41)
   return {"crop":(.27,.17,.73,.45),"waist":(.26,.17,.74,.50),"hip":(.24,.17,.76,.55)}.get(length,(.26,.17,.74,.50))
  if slot=="bottom": return {"thigh":(.28,.48,.72,.67),"knee":(.27,.48,.73,.75),"calf":(.26,.48,.74,.82),"ankle":(.25,.48,.75,.89),"floor":(.25,.48,.75,.91),"full":(.25,.48,.75,.89)}.get(length,(.25,.48,.75,.89))
  return {"shoes":(.30,.87,.70,.96),"hat":(.37,.03,.63,.13),"glasses":(.39,.12,.61,.17),"bag":(.70,.47,.88,.68),"necklace":(.39,.22,.61,.33),"earrings":(.35,.12,.65,.22),"bracelet":(.66,.54,.80,.68)}[slot]
 def compose(self,outfit_spec,width,height,background,show_layout_guides=False,**images):
  items=self._parse(outfit_spec); colors={"white":(255,255,255),"off_white":(248,248,245),"light_gray":(238,238,238)}; canvas=Image.new("RGBA",(width,height),colors[background]+(255,)); guides=ImageDraw.Draw(canvas) if show_layout_guides else None
  for slot in ("bottom","top","shoes","hat","bag","necklace","earrings","glasses","bracelet"):
   if slot not in items or images.get(slot) is None: continue
   item=items[slot] if isinstance(items[slot],dict) else {}; layout=item.get("node_layout",item); cutout=self._cutout(self._image(images[slot]))
   if cutout is None: continue
   length,coverage=layout.get("length","waist"),layout.get("coverage"); x0,y0,x1,y1=self._zone(slot,length,coverage); x0,y0,x1,y1=int(width*x0),int(height*y0),int(width*x1),int(height*y1); zw,zh=x1-x0,y1-y0
   fw={"slim":.82,"regular":.90,"loose":1,"oversized":1.08}.get(layout.get("fit","regular"),.90)
   if slot=="bottom": fw*={"slim":.84,"straight":.92,"wide":1,"baggy":1.04,"skirt":1.03}.get(layout.get("silhouette"),1)
   if slot=="top": scale=min((zw*fw)/cutout.width,(zh*{"crop":1.35,"waist":1.18,"hip":1.15}.get(length,1.18))/cutout.height)
   else: scale=min((zw*fw)/cutout.width,zh/cutout.height)
   tw,th=max(1,int(cutout.width*scale)),max(1,int(cutout.height*scale)); cutout=cutout.resize((tw,th),Image.Resampling.LANCZOS); x=x0+(zw-tw)//2; top=y1-th if slot in ("top","shoes") else y0+(zh-th)//2
   if guides: guides.rectangle((x0,y0,x1,y1),outline=(255,45,45,255),width=max(1,width//384)); guides.text((x0+4,y0+4),f"{slot}:{length}"+(f"/{coverage}" if coverage else ""),fill=(255,45,45,255))
   shadow=Image.new("RGBA",(tw,th),(0,0,0,0)); shadow.putalpha(cutout.getchannel("A").filter(ImageFilter.GaussianBlur(3)).point(lambda p:p//7)); canvas.alpha_composite(shadow,(x+3,top+5)); canvas.alpha_composite(cutout,(x,top))
  return (torch.from_numpy(np.asarray(canvas.convert("RGB")).astype(np.float32)/255).unsqueeze(0),)
NODE_CLASS_MAPPINGS={"OutfitReferenceComposer":OutfitReferenceComposer}
NODE_DISPLAY_NAME_MAPPINGS={"OutfitReferenceComposer":"Outfit Reference Composer"}
