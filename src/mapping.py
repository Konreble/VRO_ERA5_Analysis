'''
THIS FILE IS COMPLETELY AI GENERATED
Run this file to generate the map used in the report.
it is used to create the maps used in the report, however as it is not technically part of our topic,
I did not try to understand or edit this file other than making sure it works.
I considered removing it from the final python project but for reproduction of our results I decided to keep it in.
-Konrad
'''

from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# 1. Bounding Box & Viewport
BBOX = {"north": 38.5, "south": 36.5, "west": 12.25, "east": 15.75}

# Tightly focused extent on Sicily
MAIN_EXTENT = [11.8, 16.0, 36.2, 38.8]

# Context showing Mediterranean up to Germany
INSET_EXTENT = [-5.0, 22.0, 34.0, 55.0]

output_path = Path("../output/figures/study_area_map.png")
output_path.parent.mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize=(12, 9))

# -------------------------------------------------------------
# MAIN MAP (Full-canvas zoom on Sicily)
# -------------------------------------------------------------
ax_main = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax_main.set_extent(MAIN_EXTENT, crs=ccrs.PlateCarree())

# High-resolution features
ax_main.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f4f3ef")
ax_main.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#dceef8")
ax_main.add_feature(
    cfeature.COASTLINE.with_scale("10m"), edgecolor="#222222", linewidth=0.9
)
ax_main.add_feature(
    cfeature.BORDERS.with_scale("10m"),
    linestyle=":",
    edgecolor="#666666",
    linewidth=0.7,
)

# ERA5 Extraction Bounding Box
width = BBOX["east"] - BBOX["west"]
height = BBOX["north"] - BBOX["south"]

rect = mpatches.Rectangle(
    xy=(BBOX["west"], BBOX["south"]),
    width=width,
    height=height,
    fill=False,
    color="#d95f02",
    linewidth=2.5,
    linestyle="--",
    transform=ccrs.PlateCarree(),
    label="ERA5 Selection Domain",
    zorder=5,
)
ax_main.add_patch(rect)

# Gridlines
gl = ax_main.gridlines(
    draw_labels=True, dms=False, color="gray", alpha=0.3, ls=":"
)
gl.top_labels = False
gl.right_labels = False

ax_main.set_title(
    "Study Area: Sicily ERA5 Extraction Domain",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax_main.legend(loc="lower right", framealpha=0.95, fontsize=10)

# -------------------------------------------------------------
# INSET MAP (Bottom-left corner, positioned over open sea)
# -------------------------------------------------------------
# [left, bottom, width, height] in normalized figure coordinates
ax_inset = fig.add_axes([0.14, 0.16, 0.28, 0.28], projection=ccrs.PlateCarree())
ax_inset.set_extent(INSET_EXTENT, crs=ccrs.PlateCarree())

ax_inset.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e8e8e6")
ax_inset.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#cce2ec")
ax_inset.add_feature(
    cfeature.COASTLINE.with_scale("50m"), edgecolor="#444444", linewidth=0.5
)
ax_inset.add_feature(
    cfeature.BORDERS.with_scale("50m"),
    linestyle=":",
    edgecolor="#777777",
    linewidth=0.4,
)

# Reference country labels on inset
ax_inset.text(
    10.0,
    51.0,
    "Germany",
    fontsize=7.5,
    color="#333333",
    fontweight="bold",
    ha="center",
    transform=ccrs.PlateCarree(),
)
ax_inset.text(
    12.5,
    42.0,
    "Italy",
    fontsize=7.5,
    color="#333333",
    fontweight="bold",
    ha="center",
    transform=ccrs.PlateCarree(),
)

# Red highlight box showing where the main Sicily map is located
main_w = MAIN_EXTENT[1] - MAIN_EXTENT[0]
main_h = MAIN_EXTENT[3] - MAIN_EXTENT[2]
ax_inset.add_patch(
    mpatches.Rectangle(
        xy=(MAIN_EXTENT[0], MAIN_EXTENT[2]),
        width=main_w,
        height=main_h,
        fill=True,
        facecolor="#e41a1c",
        edgecolor="#990000",
        alpha=0.45,
        linewidth=1.2,
        transform=ccrs.PlateCarree(),
        zorder=10,
    )
)

# Clean border for the inset window
for spine in ax_inset.spines.values():
    spine.set_edgecolor("#333333")
    spine.set_linewidth(1.2)
ax_inset.set_title("Regional Overview", fontsize=8.5, pad=3, fontweight="bold")

fig.savefig(output_path, dpi=300, bbox_inches="tight")