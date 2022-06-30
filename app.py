import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go 
import plotly.figure_factory as ff
import plotly.express as px 
from skimage import data
from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure, ColumnDataSource, show
from bokeh.tile_providers import get_provider, Vendors
from bokeh.palettes import PRGn, RdYlGn
from bokeh.transform import linear_cmap,factor_cmap
from bokeh.layouts import row, column
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, NumeralTickFormatter
from urllib.request import urlopen
import json
import plotly.offline as py
from math import log, sqrt
from scipy.integrate import odeint


# Interactive Heatmap with Plotly
def heatmap(df):
    st.write('Figure 1: Heatmap For Female Genital Mutilation')
    trace = go.Heatmap(z=[[1, 20, 30, 50, 1], [20, 1, 60, 80, 30], [30, 60, 1, -10, 20]],
                    x=['Burkina', 'Egypt', 'Ethiopia', 'Gambia', 'Guinea'],
                    y=['Kenya', 'Mauritania', 'Nigeria'])
    data=[trace]
    layout = go.Layout(title='Heatmap For Female Genital Mutilation')
    figure = go.Figure(data=data, layout=layout)
    f2 = go.FigureWidget(figure)
    st.plotly_chart(f2)
    st.write("""
             Figure 1, above shows the heatmap of the interaction between the eight African countries 
             (Kenya, Egypt, Burkina Faso, Gambia, Guinea, Mauritiana, Nigeria and soon) that still 
             practice female genital mutilation and the data manipulation was done using Panda’s 
             library while Plotly produces the outstanding complex interactive diagram that made it 
             more unique than other python libraries used in this study. The outstanding interactive 
             features were because of the Plotly library while Pandas helps in manipulating the dataset.
             """)

# Making Chart with Plotly and pandas
def chart(df):
    st.write('Plotting 3D graph with Plotly and Pandas')
    fig = ff.create_table(df, height_constant=60)
    y = df[["FGM", "Urban", "Rural", "Poorest", "Second", "Middle", "Richest" ]]
    x = df["Countries"]
    # Make traces for graph
    trace1 = go.Bar(x=x, y=y, xaxis='x2', yaxis='y2',
                    marker=dict(color='#0099ff'),
                    name='Female Genital Mutilation with wealth quantile For<br>Per Countries')
    trace2 = go.Bar(x=x, y=y, xaxis='x2', yaxis='y2',
                    marker=dict(color='#404040'),
                    name='Female Genital Mutilation with wealth quantile Against<br>Per Countries')

    # Add trace data to figure
    fig.add_traces([trace1, trace2])

    # initialize xaxis2 and yaxis2
    fig['layout']['xaxis2'] = {}
    fig['layout']['yaxis2'] = {}

    # Edit layout for subplots
    fig.layout.yaxis.update({'domain': [0, .45]})
    fig.layout.yaxis2.update({'domain': [.6, 1]})

    # The graph's yaxis2 MUST BE anchored to the graph's xaxis2 and vice versa
    fig.layout.yaxis2.update({'anchor': 'x2'})
    fig.layout.xaxis2.update({'anchor': 'y2'})
    fig.layout.yaxis2.update({'title': 'Female Genital Mutilation with wealth quantile'})

    # Update the margins to add a title and see graph x-labels.
    fig.layout.margin.update({'t':75, 'l':50})
    fig.layout.update({'title': 'Data visualization Chart'})

    # Update the height because adding a graph vertically will interact with
    # the plot height calculated for the table
    fig.layout.update({'height':800,'width':900})

    # Plot!
    st.plotly_chart(fig)
    st.write()

# Plotting 3D graph with Plotly and Pandas
def d3d_graph(df):
    fig = px.scatter_3d(df, x="Countries", y="FGM", z="Urban", color_continuous_scale="Viridis")
    st.plotly_chart(fig)

# Heap map by combining Plotly, Seaborn and Matplotlib
def heatmap_psm(df):
    plt.xlabel('Countries')
    plt.ylabel ('Female Genital Multilation Values')
    fig = px.density_heatmap(df, x="Countries", y=["FGM","Urban","Rural","Poorest","Second","Middle","Richest"], nbinsx=30, nbinsy=30, color_continuous_scale="Viridis")
    st.plotly_chart(fig)
    
# Scatter Plot by combining Pandas, Seaborn and Matplotlib
def scatter_psm(df):
    fig = sns.pairplot(df)
    st.pyplot(fig)
    
# Plotting image with Plotly
def plot_image(df):
    img = data.astronaut()
    fig = px.imshow(img, binary_format="jpeg", binary_compression_level=0)
    st.plotly_chart(fig)

# Hexbins plots with NumPy and Matplotlib
def hexbins(df):
    # y = df[["FGM", "Urban", "Rural", "Poorest", "Second", "Middle", "Richest" ]]
    # x = df["Countries"]
    x = np.random.normal(size=(1, 1000))
    y = np.random.normal(size=(1, 1000))
    fig = plt.hexbin(x, y, gridsize=15)
    st.pyplot(fig)

# Line Plot with Pandas and Plotly
def line_plot(df):
    fig = px.line(df, x="Countries", y=["FGM","Urban","Rural","Poorest","Second","Middle","Richest"], title='Data Visualization')
    st.plotly_chart(fig)
    
# Multiple Bar chart with Pandas and Plotly
def multi_plot(df):
    fig = px.bar(df, x='Countries', y=["FGM","Urban","Rural","Poorest","Second","Middle","Richest"])
    st.plotly_chart(fig)

# Box plot with Pandas and Plotly
def box_plot(df):
    fig = px.box(df, x='Countries', y=["FGM","Urban","Rural","Poorest","Second","Middle","Richest"], points="all")
    st.plotly_chart(fig)

# Female genital mutilation visualization to show three different quartile computation
def three_quantile():
    data = [76,87,65,76,95,21,67,20]
    df = pd.DataFrame(dict(
        linear=data,
        inclusive=data,
        exclusive=data
    )).melt(var_name="quartilemethod")
    fig = px.box(df, y="value", facet_col="quartilemethod", color="quartilemethod",
    boxmode="overlay", points='all')
    fig.update_traces(quartilemethod="linear", jitter=0, col=1)
    fig.update_traces(quartilemethod="inclusive", jitter=0, col=2)
    fig.update_traces(quartilemethod="exclusive", jitter=0, col=3)
    st.plotly_chart(fig)

# Basic Funnel Plot with Plotly and Pandas
def funnel_plot(df):
    data = dict(
        FGM=[76,87,65,76,95,21,67,20],
        Countries=["Burkina", "Egypt", "Ethiopia", "Gambia", "Guinea", "Kenya", "Mauritania", "Nigeria"]
    )
    fig = px.funnel(data, x='FGM', y='Countries')
    st.plotly_chart(fig)

# Density Heatmaps with Plotly and Pandas
def density_plot(df):
    fig = px.density_heatmap(df, x="Countries", y="FGM")
    st.plotly_chart(fig)

# Interactive line graph with Plotly and Pandas
def igraph(df):
    fig = px.line(df, x='Countries', y="FGM")
    st.plotly_chart(fig)

# Interactive scattered plot with Plotly
def iscatter(df):
    fig = px.scatter(df, x="Countries", y=["FGM","Urban","Rural","Poorest","Second","Middle","Richest"])
    st.plotly_chart(fig)

# Advanced Interactive Pie Chart with Plotly
def ipie(df):
    fig2 = px.sunburst(df, path=['Countries'], values='FGM', color='Countries')
    fig2.update_layout(title_text="Female Genital Mutilation", font_size=10)
    st.plotly_chart(fig2)
    

# Map with Plotly
def map_ployly(df):
    fig = px.density_heatmap(df, x='FGM', y='Countries')
    st.plotly_chart(fig)




#Set the data source
data_file = "data_am.xlsx"

#Page configuration
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
	layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
	page_title="Dashboard",  # String or None. Strings get appended with "• Streamlit". 
	page_icon=None,  # String, anything supported by st.image, or None.
)
st.sidebar.subheader('Result')
screen = st.sidebar.selectbox("Select the visualsation",
                              ['Interactive Heatmap with Plotly', 'Making Chart with Plotly and pandas',
                               'Plotting 3D graph with Plotly and Pandas','Heap map by combining Plotly, Seaborn and Matplotlib',
                               'Scatter Plot by combining Pandas, Seaborn and Matplotlib','Plotting image with Plotly',
                               'Line Plot with Pandas and Plotly','Multiple Bar chart with Pandas and Plotly',
                               'Box plot with Pandas and Plotly',
                               'Female genital mutilation visualization to show three different quartile computation',
                               'Basic Funnel Plot with Plotly and Pandas','Density Heatmaps with Plotly and Pandas',
                               'Map with Plotly','Advanced Interactive Pie Chart with Plotly'])


df = pd.read_excel(data_file, sheet_name='Data')

if(screen == 'Interactive Heatmap with Plotly'):
    heatmap(df)
elif(screen=='Making Chart with Plotly and pandas'):
    chart(df)
elif(screen == 'Plotting 3D graph with Plotly and Pandas'):
    d3d_graph(df)
elif(screen == 'Heap map by combining Plotly, Seaborn and Matplotlib'):
    heatmap_psm(df)
elif(screen == 'Scatter Plot by combining Pandas, Seaborn and Matplotlib'):
    scatter_psm(df)
elif(screen == 'Plotting image with Plotly'):
    plot_image(df)
elif(screen == 'Line Plot with Pandas and Plotly'):
    line_plot(df)
elif(screen == 'Multiple Bar chart with Pandas and Plotly'):
    multi_plot(df)
elif(screen == 'Box plot with Pandas and Plotly'):
    box_plot(df)
elif(screen == 'Female genital mutilation visualization to show three different quartile computation'):
    three_quantile()
elif(screen == 'Basic Funnel Plot with Plotly and Pandas'):
    funnel_plot(df)
elif(screen == 'Density Heatmaps with Plotly and Pandas'):
    density_plot(df)
elif(screen == 'Map with Plotly'):
    map_ployly(df)
elif(screen == 'Advanced Interactive Pie Chart with Plotly'):
    ipie(df)
