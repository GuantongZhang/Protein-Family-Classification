from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

ea = event_accumulator.EventAccumulator("lightning_logs/version_30265038/events.out.tfevents.1743872307.v001.ib.bridges2.psc.edu.85319.0")
ea.Reload()
#protein/base_code/lightning_logs/version_30250421/events.out.tfevents.1743741900.v006.ib.bridges2.psc.edu.96668.0
#protein/base_code/lightning_logs/version_30265038/events.out.tfevents.1743872260.v001.ib.bridges2.psc.edu.85284.0
#protein/base_code/lightning_logs/version_30265038/events.out.tfevents.1743872307.v001.ib.bridges2.psc.edu.85319.0

# 提取所有指标到DataFrame
metrics = []
for tag in ea.scalars.Keys():
    for event in ea.scalars.Items(tag):
        metrics.append({
            "step": event.step,
            "epoch": event.step,  # 假设step=epoch
            "metric": tag,
            "value": event.value
        })

df = pd.DataFrame(metrics)
pivot_df = df.pivot(index="epoch", columns="metric", values="value")

# 查看val_acc按epoch的变化
print(pivot_df["val_acc"].dropna())