# Server B: Exchange-Rate

12/12 cases complete; original model vs frozen OGRR, 20 paired test masks.

|Case|Raw MSE|OGRR MSE|Raw MAE|OGRR MAE|
|---|---:|---:|---:|---:|
|Exchange-Rate/sr015_bl8/SAITS|0.30887778|0.06952805|0.46510215|0.17857883|
|Exchange-Rate/sr015_bl8/ModernTCN|0.01828842|0.02437416|0.08463333|0.10617822|
|Exchange-Rate/sr015_bl8/TimeMixerPP|0.05606569|0.01529391|0.15062530|0.07101724|
|Exchange-Rate/sr010_bl8/SAITS|0.35722456|0.09640144|0.47911396|0.20924440|
|Exchange-Rate/sr010_bl8/ModernTCN|0.01241104|0.01565397|0.07086768|0.08635732|
|Exchange-Rate/sr010_bl8/TimeMixerPP|0.04396103|0.00929021|0.13998825|0.05664983|
|Exchange-Rate/sr020_bl8/SAITS|0.40564280|0.14348072|0.50566863|0.26263498|
|Exchange-Rate/sr020_bl8/ModernTCN|0.05301985|0.07056232|0.11110237|0.15830495|
|Exchange-Rate/sr020_bl8/TimeMixerPP|0.11332730|0.05320682|0.20074770|0.10931128|
|Exchange-Rate/sr010_bl16/SAITS|0.45356714|0.25489419|0.54673374|0.35902317|
|Exchange-Rate/sr010_bl16/ModernTCN|0.19642342|0.21808424|0.19345514|0.27245693|
|Exchange-Rate/sr010_bl16/TimeMixerPP|0.27437677|0.16200969|0.32571027|0.17806199|

Negative results retained. Mask SD is not training-seed SD. Extract archive into the experiment root to merge only this dataset.
