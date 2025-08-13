from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

def visualize_matching(sec1_df, sec2_df, matches, names1, names2,
                       centroid_x_col, centroid_y_col, out_png, title_l, title_r):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    msize, tsize = 50, 10

    axes[0].scatter(sec1_df[centroid_x_col], sec1_df[centroid_y_col], s=msize, color='blue')
    for i, n in enumerate(names1):
        cid = n.split('_')[0]
        axes[0].text(sec1_df.iloc[i][centroid_x_col], sec1_df.iloc[i][centroid_y_col], cid, fontsize=tsize)
    axes[0].set_title(title_l); axes[0].set_aspect('equal')

    axes[1].scatter(sec2_df[centroid_x_col], sec2_df[centroid_y_col], s=msize, color='green')
    for i, n in enumerate(names2):
        cid = n.split('_')[0]
        axes[1].text(sec2_df.iloc[i][centroid_x_col], sec2_df.iloc[i][centroid_y_col], cid, fontsize=tsize)
    axes[1].set_title(title_r); axes[1].set_aspect('equal')

    for m in matches:
        p = m[0]; children = m[1] if isinstance(m[1], list) else [m[1]]
        i = int(p.split('_')[1])
        c1 = (sec1_df.iloc[i][centroid_x_col], sec1_df.iloc[i][centroid_y_col])
        axes[0].plot([c1[0]], [c1[1]], 'ro')
        for ch in children:
            j = int(ch.split('_')[1])
            c2 = (sec2_df.iloc[j][centroid_x_col], sec2_df.iloc[j][centroid_y_col])
            axes[1].plot([c2[0]], [c2[1]], 'ro')
            con = ConnectionPatch(xyA=c1, xyB=c2, coordsA="data", coordsB="data",
                                  axesA=axes[0], axesB=axes[1], color="red", alpha=0.6, linestyle="--")
            axes[1].add_artist(con)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

def visualize_dynamic_matching_with_plotly(sec1_df, sec2_df, matches,
                                           names1, names2,
                                           centroid_x_col, centroid_y_col,
                                           out_html, title_l, title_r):
    x1 = sec1_df[centroid_x_col].values; y1 = sec1_df[centroid_y_col].values
    x2 = sec2_df[centroid_x_col].values; y2 = sec2_df[centroid_y_col].values

    id1 = [n.split('_')[0] for n in names1]
    id2 = [n.split('_')[0] for n in names2]

    matched = []
    for m in matches:
        p = m[0]; children = m[1] if isinstance(m[1], list) else [m[1]]
        i = int(p.split('_')[1])
        for ch in children:
            j = int(ch.split('_')[1]); matched.append((i, j))

    matched_dict = {str(i): j for i, j in matched}
    matched_rev = {str(j): i for i, j in matched}
    matched_i = [i for i, _ in matched]; matched_j = [j for _, j in matched]

    fig = make_subplots(rows=1, cols=2, subplot_titles=(title_l, title_r), horizontal_spacing=0.1)

    # matched on left
    fig.add_trace(go.Scatter(x=x1[matched_i], y=y1[matched_i], mode='markers',
                             marker=dict(size=8), name=f"Matched {title_l}",
                             text=[id1[i] for i in matched_i], hoverinfo='text',
                             customdata=matched_i), row=1, col=1)
    # unmatched on left
    unmatched1 = list(set(range(len(x1))) - set(matched_i))
    fig.add_trace(go.Scatter(x=x1[unmatched1], y=y1[unmatched1], mode='markers',
                             marker=dict(size=8), name=f"Unmatched {title_l}",
                             text=[id1[i] for i in unmatched1], hoverinfo='text',
                             customdata=unmatched1), row=1, col=1)
    # matched on right
    fig.add_trace(go.Scatter(x=x2[matched_j], y=y2[matched_j], mode='markers',
                             marker=dict(size=8), name=f"Matched {title_r}",
                             text=[id2[j] for j in matched_j], hoverinfo='text',
                             customdata=matched_j), row=1, col=2)
    # unmatched on right
    unmatched2 = list(set(range(len(x2))) - set(matched_j))
    fig.add_trace(go.Scatter(x=x2[unmatched2], y=y2[unmatched2], mode='markers',
                             marker=dict(size=8), name=f"Unmatched {title_r}",
                             text=[id2[j] for j in unmatched2], hoverinfo='text',
                             customdata=unmatched2), row=1, col=2)

    fig.update_xaxes(title_text='X', row=1, col=1); fig.update_yaxes(title_text='Y', row=1, col=1)
    fig.update_xaxes(title_text='X', row=1, col=2); fig.update_yaxes(title_text='Y', row=1, col=2)
    fig.update_layout(title=f"Interactive Matching {title_l} ↔ {title_r}", hovermode='closest', showlegend=True)

    # wiring for hover lines across subplots
    js = f"""
    var matched = {json.dumps(matched_dict)};
    var matched_rev = {json.dumps(matched_rev)};
    var id1 = {json.dumps(id1)};
    var id2 = {json.dumps(id2)};
    var left_idx  = {json.dumps(matched_i)};
    var right_idx = {json.dumps(matched_j)};
    var my = document.getElementById('plot');
    function dataToPaper(x, axis) {{
      var a = my._fullLayout[axis];
      var r = a.range; var d = a.domain; var t = (x - r[0])/(r[1]-r[0]); return d[0] + t*(d[1]-d[0]);
    }}
    my.on('plotly_hover', function(data){{
      var p = data.points[0], c = p.curveNumber;
      Plotly.relayout(my, {{'shapes': [], 'annotations': []}});
      var shapes=[], ann=[];
      if (c === 0) {{
        var idx1 = p.customdata, idx2 = matched[idx1];
        if (idx2 !== undefined) {{
          idx2 = parseInt(idx2);
          var pos = right_idx.indexOf(idx2);
          var x1 = my.data[2].x[pos], y1 = my.data[2].y[pos];
          var x0p = dataToPaper(p.x, 'xaxis'), y0p = dataToPaper(p.y, 'yaxis');
          var x1p = dataToPaper(x1, 'xaxis2'), y1p = dataToPaper(y1, 'yaxis2');
          shapes.push({{type:'line', x0:x0p,y0:y0p,x1:x1p,y1:y1p, xref:'paper',yref:'paper', line:{{width:2, dash:'dot'}}}});
          ann.push({{x:0.5,y:1.12,xref:'paper',yref:'paper',showarrow:false,text:'Frame L ID '+id1[idx1]+' ↔ Frame R ID '+id2[idx2] }});
        }}
      }} else if (c === 2) {{
        var idx2 = p.customdata, idx1 = matched_rev[idx2];
        if (idx1 !== undefined) {{
          idx1 = parseInt(idx1);
          var pos = left_idx.indexOf(idx1);
          var x1 = my.data[0].x[pos], y1 = my.data[0].y[pos];
          var x0p = dataToPaper(x1, 'xaxis'), y0p = dataToPaper(y1, 'yaxis');
          var x1p = dataToPaper(p.x, 'xaxis2'), y1p = dataToPaper(p.y, 'yaxis2');
          shapes.push({{type:'line', x0:x0p,y0:y0p,x1:x1p,y1:y1p, xref:'paper',yref:'paper', line:{{width:2, dash:'dot'}}}});
          ann.push({{x:0.5,y:1.12,xref:'paper',yref:'paper',showarrow:false,text:'Frame L ID '+id1[idx1]+' ↔ Frame R ID '+id2[idx2] }});
        }}
      }}
      Plotly.relayout(my, {{'shapes': shapes, 'annotations': ann}});
    }});
    my.on('plotly_unhover', function(){{ Plotly.relayout(my, {{'shapes': [], 'annotations': []}}); }});
    """
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs='cdn', full_html=True, post_script=js, div_id='plot')
