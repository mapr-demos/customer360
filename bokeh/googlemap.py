##################
# define geo map
##################
from bokeh.models import HoverTool

def create_geomap(gmapsource):

    from bokeh.io import output_file, show
    from bokeh.models import (
        GMapPlot, GMapOptions, ColumnDataSource, Circle, LinearColorMapper, BasicTicker, ColorBar,
        DataRange1d, PanTool, WheelZoomTool, BoxSelectTool
    )

    map_options = GMapOptions(lat=37.6, lng=-119.5, map_type="roadmap", zoom=6)

    # For GMaps to function, Google requires you obtain and enable an API key:
    #
    #     https://developers.google.com/maps/documentation/javascript/get-api-key
    #
    # Replace the value below with your personal API key:
    gmapplot = GMapPlot(
        api_key="AIzaSyBYrbp34OohAHsX1cub8ZeHlMEFajv15fY",
        x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options,
        toolbar_location="right"
    )

    gmapplot.title.text = "Predicted Lifetime Value"


    max_median_house_value = housing.loc[housing['median_house_value'].idxmax()]['median_house_value']
    min_median_house_value = housing.loc[housing['median_house_value'].idxmin()]['median_house_value']

    #color_mapper = CategoricalColorMapper(factors=['hi', 'lo'], palette=[RdBu3[2], RdBu3[0]])
    #color_mapper = LogColorMapper(palette="Viridis5", low=min_median_house_value, high=max_median_house_value)
    color_mapper = LinearColorMapper(palette="Viridis5")

    # Circle glyph with relevant properties for plotting
    circle = Circle(x="lon", y="lat", size="size", fill_color={'field': 'color', 'transform': color_mapper},
                    fill_alpha=0.5, line_color=None)
    gmapplot.add_glyph(gmapsource, circle)

    # Information to be displayed when hovering over glyphs
    gmaphover = HoverTool(
        tooltips=[
            ("(Lat,Long)", "(@lat, @lon)"),
            ("Income", "@size"),
            ("Predicted Lifetime Value", "@color"),
        ]
    )

    # Legend info
    factors = ["$250k", "$500k", "$750k", "$1M", "1.25M"]
    x = [0] * 5
    y = factors


    gmapplot.add_tools(PanTool(), WheelZoomTool(), gmaphover)

    #show(gmapplot)
    from bokeh.plotting import figure
    from bokeh.palettes import Viridis5 as palette
    palette.reverse
    colors = []
    for income in housing.median_income.tolist():
        if income <= 16:
            colors.append(palette[0])
        elif 16 < income <= 17:
            colors.append(palette[1])
        elif 17 < income <= 18:
            colors.append(palette[2])
        elif 18 < income <= 19:
            colors.append(palette[3])
        elif income > 26:
            colors.append(palette[4])
        else:
            colors.append("#F3F1ED")

    p = figure(plot_width=70, toolbar_location=None, y_range=factors)
    p.rect(x, y, color=palette, width=2, height=1)
    p.xaxis.major_label_text_color = None
    p.xaxis.major_tick_line_color = None
    p.xaxis.minor_tick_line_color = None

    return {'gmapplot':gmapplot,'p':p}