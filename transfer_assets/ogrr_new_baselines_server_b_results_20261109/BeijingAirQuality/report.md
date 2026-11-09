# Server B: BeijingAirQuality

12/12 cases complete; original model vs frozen OGRR, 20 paired test masks.

|Case|Raw MSE|OGRR MSE|Raw MAE|OGRR MAE|
|---|---:|---:|---:|---:|
|BeijingAirQuality/sr015_bl8/SAITS|0.09804878|0.09394638|0.16400911|0.16075015|
|BeijingAirQuality/sr015_bl8/ModernTCN|0.21624333|0.10623430|0.27456822|0.17510873|
|BeijingAirQuality/sr015_bl8/TimeMixerPP|0.22282618|0.10576892|0.28667408|0.17300294|
|BeijingAirQuality/sr010_bl8/SAITS|0.08483442|0.08002869|0.14888496|0.14475972|
|BeijingAirQuality/sr010_bl8/ModernTCN|0.13668144|0.08440887|0.21590864|0.15313027|
|BeijingAirQuality/sr010_bl8/TimeMixerPP|0.18906638|0.08737175|0.26145481|0.15438775|
|BeijingAirQuality/sr020_bl8/SAITS|0.11915369|0.11469816|0.18442223|0.18123440|
|BeijingAirQuality/sr020_bl8/ModernTCN|0.38078699|0.14485465|0.36691861|0.20435013|
|BeijingAirQuality/sr020_bl8/TimeMixerPP|0.26577278|0.13442637|0.31389965|0.19845710|
|BeijingAirQuality/sr010_bl16/SAITS|0.13538882|0.13116876|0.19623343|0.19370114|
|BeijingAirQuality/sr010_bl16/ModernTCN|0.66876149|0.18592678|0.51035914|0.22552341|
|BeijingAirQuality/sr010_bl16/TimeMixerPP|0.33670107|0.15311375|0.36079197|0.21182991|

Negative results retained. Mask SD is not training-seed SD. Extract archive into the experiment root to merge only this dataset.
