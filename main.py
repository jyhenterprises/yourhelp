import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Scenario assumptions (edit these)
# ----------------------------
calls_per_day = 10
days_per_month = 30
billed_minutes_per_call = 4          # telephony billing often rounds up to full minutes
sms_segments_per_call = 2            # example: short summary + link

# ----------------------------
# Rates (USD) used in the workflow (edit if your rates differ)
# ----------------------------
rate_local_number_monthly = 1.15
rate_inbound_answered_in_app_per_min = 0.01165
rate_voice_ai_engine_per_min = 0.06
rate_call_recording_per_min = 0.0025
rate_call_transcription_per_min = 0.024
rate_recording_storage_per_min_per_month = 0.0005
rate_sms_per_segment = 0.00747

# ----------------------------
# Derived volumes
# ----------------------------
calls_per_month = calls_per_day * days_per_month
billed_minutes_per_month = calls_per_month * billed_minutes_per_call
sms_segments_per_month = calls_per_month * sms_segments_per_call

# ----------------------------
# Cost model
# ----------------------------
items = [
    ("Local telephone number monthly rental (LeadConnector Phone System)",
     rate_local_number_monthly),
    ("Inbound call minutes answered in GoHighLevel application (LeadConnector Phone System)",
     billed_minutes_per_month * rate_inbound_answered_in_app_per_min),
    ("Voice Artificial Intelligence voice engine (Pay-Per-Use)",
     billed_minutes_per_month * rate_voice_ai_engine_per_min),
    ("Call recording (LeadConnector Phone System)",
     billed_minutes_per_month * rate_call_recording_per_min),
    ("Call transcription to text (Voice Intelligence add-on)",
     billed_minutes_per_month * rate_call_transcription_per_min),
    ("Call recording storage for one month (LeadConnector Phone System)",
     billed_minutes_per_month * rate_recording_storage_per_min_per_month),
    ("Text message sending (Short Message Service message segments)",
     sms_segments_per_month * rate_sms_per_segment),
]

df = pd.DataFrame(items, columns=["Service (full name)", "Estimated monthly cost (USD)"])
total = df["Estimated monthly cost (USD)"].sum()
df["Percent of total"] = df["Estimated monthly cost (USD)"] / total * 100
df = df.sort_values("Estimated monthly cost (USD)", ascending=False).reset_index(drop=True)

# ----------------------------
# Chart
# ----------------------------
min_pct_label = 2.0  # hide tiny-slice percentage labels under this threshold

def autopct_if_big(pct: float) -> str:
    return f"{pct:.1f}%" if pct >= min_pct_label else ""

fig, ax = plt.subplots(figsize=(10, 7))

wedges, _, _ = ax.pie(
    df["Estimated monthly cost (USD)"],
    labels=None,                 # no wedge labels to avoid collisions
    autopct=autopct_if_big,
    pctdistance=0.70,
    startangle=90,
    textprops={"fontsize": 9}
)

# Donut hole (improves readability)
centre_circle = plt.Circle((0, 0), 0.55, fc="white")
ax.add_artist(centre_circle)

ax.set_title(
    "Fractional cost by service (donut chart)\n"
    f"calls/day={calls_per_day}, billed_minutes/call={billed_minutes_per_call}, sms_segments/call={sms_segments_per_call}"
)

# Legend
legend_labels = [
    f"{name}\n${cost:,.2f} ({pct:.1f}%)"
    for name, cost, pct in zip(
        df["Service (full name)"],
        df["Estimated monthly cost (USD)"],
        df["Percent of total"]
    )
]
ax.legend(
    wedges,
    legend_labels,
    title="Services",
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.0,
    fontsize=9,
    title_fontsize=10
)

# ----------------------------
# Add centered "Total Cost" text on the figure (not the axes)
# ----------------------------
fig.text(
    0.5, 0.04,                         # x=50% (center), y=4% from bottom
    f"Total Cost (excluding language model token charges and carrier pass-through text fees): ${total:,.2f}",
    ha="center", va="center", fontsize=11
)

ax.set_aspect("equal")

# Leave room at bottom for the centered total-cost text and at right for legend
plt.tight_layout(rect=[0.0, 0.08, 0.78, 1.0])
plt.show()
