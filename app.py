import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.graph_objects as go 
import plotly.figure_factory as ff
import plotly.express as px 


# Interactive Heatmap with Plotly
def heatmap(df):
    st.write('Heatmap For Female Genital Mutilation')
    trace = go.Heatmap(z=[[1, 20, 30, 50, 1], [20, 1, 60, 80, 30], [30, 60, 1, -10, 20]],
                    x=['Burkina', 'Egypt', 'Ethiopia', 'Gambia', 'Guinea'],
                    y=['Kenya', 'Mauritania', 'Nigeria'])
    data=[trace]
    layout = go.Layout(title='Heatmap For Female Genital Mutilation')
    figure = go.Figure(data=data, layout=layout)
    f2 = go.FigureWidget(figure)
    st.plotly_chart(f2)
    st.write("""
             The heatmap of the interaction between the eight African countries 
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
    st.write("""
             The Chart above using libraries like Plotly and Pandas. 
             The Chart visualizes female genital mutilation with wealth quantile per country and female genital 
             mutilation with wealth quantile per country. Based on the analysis, using the chart, 
             we could see that the highest rate in Mauritania to the least countries having half of the 
             countries were for and against the female genital mutilation. The countries and wealth quantile 
             variables were visualized using the Chart by the data manipulation using both the Pandas and Plotly 
             libraries but Pandas offers efficient operation for manipulating data while Plotly produces the outstanding visualization.
             """)

# Plotting 3D graph with Plotly and Pandas
def d3d_graph(df):
    st.write('Plotting 3D graph with Plotly and Pandas')
    fig = px.scatter_3d(df, x="Countries", y="FGM", z="Urban", color_continuous_scale="Viridis")
    st.plotly_chart(fig)
    st.write(""""
             The 3D graphical illustration of female genital mutilation (FGM), the urban and countries like Burkina Faso, 
             Egypt, Ethiopia, Gambia, Guinea, Kenya, and Mauritania with the combination of python libraries like Plotly and Pandas.
             """)

# Heap map by combining Plotly, Seaborn and Matplotlib
def heatmap_psm(df):
    st.write('Heap map by combining Plotly, Seaborn and Matplotlib')
    plt.xlabel('Countries')
    plt.ylabel ('Female Genital Multilation Values')
    fig = px.density_heatmap(df, x="Countries", y=["FGM","Urban","Rural","Poorest","Second","Middle","Richest"], nbinsx=30, nbinsy=30, color_continuous_scale="Viridis")
    st.plotly_chart(fig)
    st.write("""
              The Heat map visualization of African countries with cases of female genital mutilation using different 
              python libraries like Plotly, Seaborn and Matplotlib to form a complex interactive diagram. 
              The Heat map shows the interrelationship in the data by manipulating the libraries and the Matplotlib 
              brings out the unique customization in the diagram.
             """)
    
# Scatter Plot by combining Pandas, Seaborn and Matplotlib
def scatter_psm(df):
    st.write('Scatter Plot matrix by combining Pandas, Seaborn and Matplotlib')
    fig = sns.pairplot(df)
    st.pyplot(fig)
    st.write("""
              The scatter plot matrix by combining different python libraries like Pandas, 
              Seaborn and Matplotlib. The scatter plot help to explore the relationship between 
              the wealth quantile variables (such as urban, rural, poorest, second, middle and richest) 
              and the female genital mutilation (FGM). The Seaborn was used for the data analysis via scatterplot, 
              Matplotlib was used for the customization and Pandas for the data manipulation.
             """)
    
# Hexbins plots with NumPy and Matplotlib
def hexbins(df):
    st.write('Hexbins plots with NumPy and Matplotlib')
    x = np.random.normal(size=(1, 1000))
    y = np.random.normal(size=(1, 1000))
    fig = plt.hexbin(x, y, gridsize=15)
    st.pyplot(fig)
    st.write("""
              the visualization of a scattered plot of two numeric data by Hexbins using two different 
              python libraries such as NumPy and Matplotlib. The Hexbins plots are an improvement on the 
              usually scattered plots and give a better visualization pattern. 
              The NumPy library is suitable for an integer and does a great manipulation of the dataset in figure 7 
              while Matplotlib produces the visualization with its unique customization function.
             """)

# Line Plot with Pandas and Plotly
def line_plot(df):
    st.write('Interactive Line Plots with Pandas and Plotly')
    fig = px.line(df, x="Countries", y=["FGM","Urban","Rural","Poorest","Second","Middle","Richest"], title='Data Visualization')
    st.plotly_chart(fig)
    st.write("""
             The interactive line graph indicating the connection of each of each Africa countries with 
             female genital mutilation. It shows the rate of female genital mutilation in each of the countries. 
             Based on the analysis, using the line plot we could conclude each line represent the variables, 
             between Nigeria and Kenya have the lowest female genital mutilation but it is most affected in the rural and the poorest countries with Mauritania, Egypt and Guinea having the highest in the displayed diagram. The variable key was indicated by the graph to show what the lines represent. The data manipulation was done by pandas while the visualisation was done by Plotly.

             """)
    
# Multiple Bar chart with Pandas and Plotly
def multi_plot(df):
    st.write('Interactive Multiple Bar chart with Pandas and Plotly')
    fig = px.bar(df, x='Countries', y=["FGM","Urban","Rural","Poorest","Second","Middle","Richest"])
    st.plotly_chart(fig)
    st.write("""The interactive multiple Bar chart that visualizes eight African countries with cases of genital 
             mutilation using Pandas and Plotly libraries.
             """)

# Box plot with Pandas and Plotly
def box_plot(df):
    st.write('')
    fig = px.box(df, x='Countries', y=["FGM","Urban","Rural","Poorest","Second","Middle","Richest"], points="all")
    st.plotly_chart(fig)
    st.write("""
             the box plot that visualizes eight African countries with female genital mutilation 
             cases using Python libraries like Pandas and Plotly. Box plots also help to identify outliers in the dataset.
             """)

# Female genital mutilation visualization to show three different quartile computation
def three_quantile():
    st.write('Female genital mutilation visualization to show three different quartile computation')
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
    st.write("""
             The difference between three quartile computation methods such as linear, 
             inclusive and exclusive with combined python libraries of Pandas and Plotly
             to visualize the female genital mutilation (FGM).
             """)

# Basic Funnel Plot with Plotly and Pandas
def funnel_plot(df):
    st.write('Basic Funnel Plot with Plotly and Pandas')
    data = dict(
        FGM=[76,87,65,76,95,21,67,20],
        Countries=["Burkina", "Egypt", "Ethiopia", "Gambia", "Guinea", "Kenya", "Mauritania", "Nigeria"]
    )
    fig = px.funnel(data, x='FGM', y='Countries')
    st.plotly_chart(fig)
    st.write("""
              The visualization of the countries with cases of female genital mutilation with a Basic funnel plot using
              Plotly and Pandas libraries. Using the Funnel plot we could state the number of the highest and lowest country, 
              with this we can be able to interpret the results of which countries has the highest and lowest range.
             """)

# Density Heatmaps with Plotly and Pandas
def density_plot(df):
    st.write('Density Heatmaps with Plotly and Pandas')
    fig = px.density_heatmap(df, x="Countries", y="FGM")
    st.plotly_chart(fig)
    st.write("""The countries with cases of female genital mutilation (FGM) with
             density heatmaps plot using a combination of python libraries like Plotly and Pandas. 
             Plotly produces an outstanding interactive diagram.
             """)

# Interactive line graph with Plotly and Pandas
def igraph(df):
    st.write('Interactive line graph with Plotly and Pandas')
    fig = px.line(df, x='Countries', y="FGM")
    st.plotly_chart(fig)
    st.write("""
            An interactive line graph indicating the connection of each African country with 
            female genital mutilation. It shows the rate of female genital mutilation in each of the countries. 
            The visualization was done by Plotly library.
             """)

# Interactive scattered plot with Plotly
def iscatter(df):
    st.write('Interactive scattered plot with Plotly')
    fig = px.scatter(df, x="Countries", y=["FGM","Urban","Rural","Poorest","Second","Middle","Richest"])
    st.plotly_chart(fig)
    st.write("""
            The interactive scatter plot of each country practising female genital mutilation with their 
            respective wealth quantile (Urban, Rural, Poorest, Second, Middle and Richest) using Plotly. 
            Recall that Seaborn cannot do the same scatter plot without combining it with Matplotlib and 
            this makes Plotly more outstanding than other libraries under this study.
             """)

# Advanced Interactive Pie Chart with Plotly
def ipie(df):
    st.write('Advanced Interactive Pie Chart with Plotly')
    fig2 = px.sunburst(df, path=['Countries'], values='FGM', color='Countries')
    fig2.update_layout(title_text="Female Genital Mutilation", font_size=10)
    st.plotly_chart(fig2)
    st.write("""the advanced Pie chart that reveals the interaction between female genital mutilation and
             the eight countries that practised it. The interactive visualization was done by Plotly library with unique colours.
             """)
    

# Map with Plotly
def map_ployly(df):
    st.write('')
    fig = px.density_heatmap(df, x='FGM', y='Countries')
    st.plotly_chart(fig)
    st.write("""
             the interaction between female genital mutilation and the eight countries practising using the Plotly library and 
             this shows that plotly is not limited in interactivity and requires simple syntax to produce outstanding visualisation performance.
             """)




#Set the data source
data_file = "data.csv"

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
                               'Scatter Plot by combining Pandas, Seaborn and Matplotlib',
                               'Line Plot with Pandas and Plotly','Multiple Bar chart with Pandas and Plotly',
                               'Box plot with Pandas and Plotly',
                               'Female genital mutilation visualization to show three different quartile computation',
                               'Basic Funnel Plot with Plotly and Pandas','Density Heatmaps with Plotly and Pandas',
                               'Map with Plotly','Advanced Interactive Pie Chart with Plotly'])


df = pd.read_csv(data_file)

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
# elif(screen == 'Plotting image with Plotly'):
#     plot_image(df)
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
