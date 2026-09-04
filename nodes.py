import json
import numpy as np
import torch
from PIL import Image

SLOTS=("top","bottom","shoes","hat","bag","glasses","necklace","earrings","bracelet")
ANCHORS={"top":(.50,.52,.45),"bottom":(.50,.43,.39),"shoes":(.50,.82,.35),"hat":(.50,.035,.28),"bag":(.77,.55,.20),"glasses":(.50,.135,.23),"necklace":(.50,.255,.22),"earrings":(.50,.145,.24),"bracelet":(.72,.61,.14)}
class OutfitReferenceComposer:
 @classmethod
 def INPUT_TYPES(cls): return {"required":{"outfit_spec":("STRING",{"multiline":True,"default":"{\n  \"items\": {}\n}"}),"width":("INT",{"default":768,"min":256,"max":2048,"step":64}),"height":("INT",{"default":1024,"min":256,"max":2048,"step":64}),"background":(["off_white","white","light_gray"],)},"optional":{s:("IMAGE",) for s in SLOTS}}
 RETURN_TYPES=("IMAGE",); RETURN_NAMES=("outfit_reference",); FUNCTION="compose"; CATEGORY="Outfit Reference"
 def compose(self,outfit_spec,width,height,background,**images):
  items=json.loads(outfit_spec).get("items",{}); bg={"white":(255,255,255),"off_white":(248,248,245),"light_gray":(238,238,238)}[background]; canvas=Image.new("RGBA",(width,height),bg+(255,))
  for slot in ("bottom","top","shoes","hat","bag","necklace","earrings","glasses","bracelet"):
   if slot not in items or images.get(slot) is None: continue
   a=np.clip(images[slot][0].detach().cpu().numpy()*255,0,255).astype(np.uint8); im=Image.fromarray(a).convert("RGBA"); alpha=np.where(np.min(a,axis=2)>244,0,255).astype(np.uint8); im.putalpha(Image.fromarray(alpha)); box=im.getbbox()
   if not box: continue
   im=im.crop(box); meta=items[slot].get("node_layout",items[slot]); cx,y,scale=ANCHORS[slot]; fit=meta.get("fit","regular"); scale*={"slim":.88,"regular":1,"loose":1.1,"oversized":1.2}.get(fit,1); w=int(width*scale); h=int(im.height*w/im.width); length=meta.get("length","waist")
   if slot=="top": y={"crop":.45,"waist":.52,"hip":.59}.get(length,.52); h=int(h*{"crop":.84,"waist":1,"hip":1.12}.get(length,1)); y-=h/height
   if slot=="bottom": h=int(h*{"thigh":.62,"knee":.75,"calf":.88,"ankle":1,"floor":1.05,"full":1}.get(length,1)); scale*={"slim":.86,"straight":.95,"wide":1.1,"baggy":1.24}.get(meta.get("silhouette"),1); w=int(width*scale); h=int(im.height*w/im.width)
   im=im.resize((w,h),Image.Resampling.LANCZOS); x=int(width*cx-w/2); top=int(height*y); x=int(width*.73) if slot=="bag" else x; canvas.alpha_composite(im,(x,top))
  out=torch.from_numpy(np.asarray(canvas.convert("RGB")).astype(np.float32)/255).unsqueeze(0); return (out,)
NODE_CLASS_MAPPINGS={"OutfitReferenceComposer":OutfitReferenceComposer}
NODE_DISPLAY_NAME_MAPPINGS={"OutfitReferenceComposer":"Outfit Reference Composer"}
