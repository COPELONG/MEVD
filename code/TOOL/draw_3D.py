import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Bar3D

data = pd.DataFrame({
    'evaluate': [
        'Acc', 'F1', 'Recall', 'Precision',
        'Acc', 'F1', 'Recall', 'Precision',
        'Acc', 'F1', 'Recall', 'Precision',
        'Acc', 'F1', 'Recall', 'Precision',
        'Acc', 'F1', 'Recall', 'Precision',
    ],
    'model': [
        'DR-GCN', 'DR-GCN', 'DR-GCN', 'DR-GCN',
        'TMP', 'TMP', 'TMP', 'TMP',
        'blstm-att', 'blstm-att', 'blstm-att', 'blstm-att',
        'bigru-att', 'bigru-att', 'bigru-att', 'bigru-att',
        'CGE', 'CGE', 'CGE', 'CGE',
    ],
    'value': [
        0.6834, 0.6632, 0.6782, 0.6489,
        0.7461, 0.7410, 0.7432, 0.7389,
        0.8760, 0.8796, 0.9059, 0.8548,
        0.8547, 0.8595, 0.8888, 0.8320,
        0.8321, 0.8213, 0.8229, 0.8197,
    ]
})

# 设置模型对应的颜色
model_color_map = {
    'DR-GCN': '#313695',  # Alizarin Crimson
    'TMP': '#008080',     # Teal
    'blstm-att': '#FF4500',  # Orange Red
    'bigru-att': '#800080',  # Purple
    'CGE': '#1E90FF',    # Dodger Blue
}

x_name = list(set(data['evaluate']))
y_name = list(set(data['model']))

data_xyz = []
for i in range(len(data)):
    x = x_name.index(data.iloc[i, 0])
    y = y_name.index(data.iloc[i, 1])
    z = data.iloc[i, 2]
    data_xyz.append([x, y, z])

bar3d = (
    Bar3D(init_opts=opts.InitOpts(width="1300px", height="800px"))
    .add(
        "",
        data_xyz,
        xaxis3d_opts=opts.Axis3DOpts(type_="category", data=x_name),
        yaxis3d_opts=opts.Axis3DOpts(type_="category", data=y_name),
        zaxis3d_opts=opts.Axis3DOpts(type_="value"),
        grid3d_opts=opts.Grid3DOpts(width=100, depth=100),
        itemstyle_opts=opts.ItemStyleOpts(
            color=lambda params: model_color_map[data.iloc[params.dataIndex, 1]]
        ),  # 根据模型选择颜色
    )
    .set_global_opts(
        visualmap_opts=opts.VisualMapOpts(
            max_=0.965, min_=0.6, range_color=list(model_color_map.values())
        ),
        title_opts=opts.TitleOpts(title="无限循环漏洞模型性能比较"),
    )
)

bar3d.render("infinite.html")
