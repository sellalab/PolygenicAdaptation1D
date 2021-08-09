import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
import numpy as np
import os
import copy
import math

default_plot_specs_all = dict()
default_plot_specs_all['fsize'] = (11, 4)  # (22, 5)
default_plot_specs_all['linewidth'] = 1.5
#default_plot_specs_all['a_lab_size'] = 20
default_plot_specs_all['axis_font'] = {'fontname': 'Arial', 'size': '18'}  # {'fontname':'Arial', 'size':'14'}
default_plot_specs_all['ticksize'] = 18
default_plot_specs_all['nxticks'] = 0
default_plot_specs_all['nyticks'] = 0
default_plot_specs_all['dpi'] = 180
default_plot_specs_all['undertext_font'] = {'color': 'black', 'weight': 'roman', 'size': 'x-small'}
default_plot_specs_all['x_scale'] = 0
default_plot_specs_all['y_scale'] = 0.05
default_plot_specs_all['marker_size'] = None
default_plot_specs_all['cap_size'] = 7
default_plot_specs_all['marker_style'] = 'o'
default_plot_specs_all['linestyle'] = '-'
default_plot_specs_all['vlinestyle'] = '--'
default_plot_specs_all['vlinecolor'] = 'indianred'
default_plot_specs_all['vlineswidth'] = 1
default_plot_specs_all['text_color'] = 'green'
default_plot_specs_all['text_loc'] = [0.8,0.6]
default_plot_specs_all['text_size'] = 15
default_plot_specs_all['yrotation'] = 'vertical'
default_plot_specs_all['ylabelspace'] = 0
default_plot_specs_all['xlog'] = False
default_plot_specs_all['ylog'] = False
default_plot_specs_all['xshade_color'] = 'yellow'
default_plot_specs_all['xshade'] = []


line_styles = ['-','-.','--',':']
marker_styles =['o','*','D','s','+',',', '-', '.']
for i in range(20):
    line_styles+=line_styles
    marker_styles+=marker_styles

def find_nearest_above(array,value):
    idx = np.searchsorted(array, value, side="left")
    return array[idx]

def find_nearest_below(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0:
        return array[idx-1]
    else:
        return array[idx]


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def save_fig(fig, save_dir, lgd=None):

    if lgd is None:
        lgd = False
    if lgd:
        axes = fig.get_axes()
        lgd = axes[0].legend()
        #print 'lgd ', save_dir
        fig.savefig(save_dir, bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        #print 'no lgd ', save_dir
        fig.savefig(save_dir, bbox_inches='tight')  # ,bbox_extra_artists=(lgd,))

    plt.close('all')


def plot_many_y(x, y, yer=None, xlabel = None, ylabel = None, ynames = None, label = None, domain=None,
                yrange = None, undertext =None, savedir = None, marker=None, markerstyles=None, plotspecs = None, groupings=None,
                groupings_labels_within = None, vlines = None, legend_title=None, n_legend_columns=None, text=None, linestyles=None,
                colors=None, save=None):
    """
    Plots multiple y against x, on a single set of axes. Saves the plot.

    Parameters:
        x: (list) A list [x_1,x_2,...] with every x_i a list of x values
        y: (list) A list [y_1, y_2...] with every y_i a list of y values
        yer: (list) A list [yer_1, yer_2,...] with every yer_i a list of the standard error on y_i
        marker: (boolean) If true, use markers instead of line
        linestyles: (list) A list of the styles to be used for the lines (e.g. ['-','-.','--',':'])
        markerstyles: (list) A list of the styles to be used for the markers (e.g. 'o','*','D','s','+',',', '-', '.'])
        colors: (list) A list of the colours to be used for the lines (e.g. 'blue','green')
        xlabel: (string) The name of the x-axis
        ylabel: (float) The name of the y-axis
        ynames: (list) A list of the names of the things being plotted
        legend_title: (string) A label for the legend
        n_legend_columns: (int) Number of columns in the legend
        label: (string) A label for the plot
        domain: (list) Form [xlow, xhigh] - the domain of the plot
        yrange: (list) Form [ylow, yhigh] - the range of the plot
        undertext: (list) List of strings to be placed beneath the plot
        text: (string) A string to be printed on the plot
        savedir: (string) The name of the directory to save the graph in
        groupings: (list of sets) each set has the index of plots to be the same colour
        groupings_labels_within: (list of strings) a list of the labels of the i-th element
                                of each group (length should be the lenghth of the longest set in groupings)
        vlines: (list of floats or float) Values at which a vertical line is to be plotted
        plotspecs: (dict) A dictionary to set the plot specifications if they differ from default -
            Possible keys:
                xshade: (list) Form [xlow, xhigh] if there should be shading between xlow and xhigh
                xshade_color: (string) colour of the shading (e.g. 'red')
                xlog: (bool) if true plots x-axis on log scale
                ylog: (bool) if true plots y-axis on log scale
                fsize: (tup) Gives dimensions of the plot (e.g. (11,4))
                linewidth: (float) Width of the lines in the plot
                marker_size: (float) Size of marker
                axis_font: (dict) (e.g. {'fontname': 'Arial', 'size': '18'})
                ticksize: (float) Size of the ticks
                yrotation: (float) Rotation of y-label (e.g. 0 means horizontal)
                ylabelspace: (float) Factor of font size that you want space between ylabel and y-axis.
                nxticks: (int) Number of xticks
                nyticks: (int) Number of yticks
                dpi: (float) Quality of the image
                undertext_font: (dict) Font of undertext
                    (e.g. {'color': 'black', 'weight': 'roman', 'size': 'x-small'})
                text_color: (string) Colour of the text on the plot (e.g. 'green')
                text_loc: (list) [x_pos, y_pos] of the text (e.g. [0.8,0.5])
                text_size: (float) size of the text
                x_scale: (float) The x-axis is (1+2x_scale)*(xmax -xmin) wide, if domain is None
                y_scale: (float) The y-axis is (1+2y_scale)*(ymax -ymin) high, if domain is None
                legend_font: (dict) Size and font of legend writing (e.g. {'size': 8})
                legend_anchor: (string) The point on the legend that's anchored (e.g. upper right)
                legend_loc: (tup) Point on plot where the legend is anchored (e.g. (0.98, -0.1))
                vlinestyle: (string) The style of vertical line
                vlinecolor: (string) The colour of vertical lines
                vlineswidth: (float) Width of the vline

    """
    if save is None:
        save = True
    if savedir is None:
        save_dir = os.getcwd()
    else:
        save_dir = savedir
    if marker is None:
        marker = False
    if vlines is None:
        vlines = []
    if isinstance(vlines, float):
        vlines = [vlines]
    if n_legend_columns is None:
        n_legend_columns = 1

    if markerstyles is None:
        my_marker_styles = [st for st in marker_styles]
    else:
        my_marker_styles = [st for st in markerstyles]
    if groupings_labels_within is None:
        groupings_labels_within = False

    if linestyles is None:
        my_line_styles = [ls for ls in line_styles]
    else:
        my_line_styles = [ls for ls in linestyles]


    #in case linestyle -- comes up
    dashes = (10, 25)
    dashes = [20,55]
    dashes = [40, 40]
    dashes = [5, 5]
    dash_width_factor = 2
    dash_width_factor = 1.5

    number_y = len(y)

    if groupings is None:
        grouped = False
        #print(["hi" for _ in range(number_y_num)])
        groupings = [{ii} for ii in range(number_y)]
    else:
        grouped = True

    # Make sure all the elements are in a colour grouping
    if grouped:
        extra_group = set()
        for i in range(number_y):
            in_a_group = False
            for seti in groupings:
                for el in seti:
                    if i == el:
                        if not in_a_group:
                            in_a_group = True
                        #else:
                            #print el, ' in two colour groups'
            if not in_a_group:
                extra_group.add(i)

    if len(groupings) == 1:
        if ynames is not None:
            if len(ynames) == number_y:
                grouped = False


    default_plot_specs = copy.deepcopy(default_plot_specs_all)
    default_plot_specs['legend_font'] = {'size': 8}
    default_plot_specs['legend_anchor'] = 'upper right'
    default_plot_specs['legend_loc'] = (0.98, -0.1)

    if marker:
        default_plot_specs['x_scale'] = 0.05
    else:
        default_plot_specs['x_scale'] = 0

    text_heights = [-0.023, -0.069, -0.115,-0.161]

    if plotspecs is not None:
        for stat in list(default_plot_specs.keys()):
            if stat in plotspecs:
                default_plot_specs[stat] = plotspecs[stat]

    the_label = ''

    if domain is not None:
        xlow = domain[0]
        xhigh = domain[1]
        for ii in range(number_y):
            klow = x[ii].index(find_nearest(x[ii],xlow))
            khigh = x[ii].index(find_nearest(x[ii], xhigh))
            #khigh = x[ii].index(find_nearest_above(x[ii], xhigh))
            x[ii] = x[ii][klow:khigh]
            y[ii] = y[ii][klow:khigh]
            if yer:
                yer[ii] = yer[ii][klow:khigh]
    if yrange is not None:
        ylow = yrange[0]
        yhigh = yrange[1]
    if xlabel is None:
        x_label = ''
    else:
        x_label = xlabel
    if ylabel is None:
        y_label = ''
        the_label = 'y_' +str(number_y) +'_'
    else:
        y_label = ylabel
        the_label += y_label[:4] +'_'
    if ynames is None:
        y_names = []
    else:
        y_names = ynames
    if label is None:
        the_label = the_label + 'vs_' +x_label
    else:
        the_label = label

    under_text = []
    if undertext is not None:
        under_text = undertext[:]

    if marker:
        rcParams['legend.numpoints'] = 1

    plt.clf()

    fig = plt.figure(figsize=default_plot_specs['fsize'], dpi=default_plot_specs['dpi'])
    ax_1 = fig.add_subplot(111)

    if default_plot_specs['xlog']:
        ax_1.set_xscale('log')
    if default_plot_specs['ylog']:
        ax_1.set_yscale('log')

    if grouped:
        mycolors = cm.rainbow(np.linspace(0, 1, len(groupings)))
    else:
        mycolors = cm.rainbow(np.linspace(0, 1, number_y))
    color_dict = dict()
    line_style_dict = dict()
    marker_style_dict = dict()


    ynames_dict = dict()
    custom_legend_entries_dict = dict()
    display_leg_numbers = []

    add_dummy_ynames = False
    if ynames is not None:
        if len(ynames) == len(groupings):
            if len(groupings) != len(y):
            # if only the first element of each group is named
                add_dummy_ynames = True
                if not groupings_labels_within:
                    display_leg_numbers = [kk for kk in range(len(ynames))]
            elif not groupings_labels_within:
                display_leg_numbers = [kk for kk in range(len(ynames))]
        elif not groupings_labels_within:
            display_leg_numbers = [kk for kk in range(len(ynames))]


    for seti, jj in zip(groupings, range(len(groupings))):
        for k,ii in zip(sorted(list(seti)), range(len(seti))):
            #jj is the group number
            #ii is the number within the set
            #k is the number in the ylist
            if colors is None:
                if grouped:
                    color_dict[k] = mycolors[jj]
                else:
                    color_dict[k] = mycolors[k]

            else:
                if grouped:
                    color_dict[k] = colors[jj]
                else:
                    color_dict[k] = colors[k]
            if grouped:
                marker_style_dict[k] = my_marker_styles[ii]
                line_style_dict[k] = my_line_styles[ii]
            else:
                # print(k)
                # print(markerstyles)
                if markerstyles is None:
                    marker_style_dict[k] = default_plot_specs['marker_style']
                else:
                    marker_style_dict[k] = markerstyles[k]
                if linestyles is None:
                    line_style_dict[k] = default_plot_specs['linestyle']
                else:
                    line_style_dict[k] = linestyles[k]
            if add_dummy_ynames:
                if ii == 0:  # if the first in the set
                    ynames_dict[k] = ynames[jj]
                else:
                    ynames_dict[k] = 'dummy'



            if groupings_labels_within:

                if ii == 0:
                    display_leg_numbers.append(k)

                # Create custom artists
                if marker:
                    markstyli = marker_style_dict[k]
                    style = line_style_dict[k]
                    if markstyli and not style:
                        capsizi = default_plot_specs['cap_size']
                    else:
                        capsizi = None
                    if line_style_dict[k] == '--':
                        custom_legend_entries_dict[ii] = plt.Line2D((0, 1), (0, 0), color='k', marker=markstyli,
                                                                    markersize=default_plot_specs['marker_size'],
                                                                    dashes=dashes)
                    else:
                        custom_legend_entries_dict[ii] = plt.Line2D((0, 1), (0, 0), color='k', marker=markstyli,
                                                                    markersize=default_plot_specs['marker_size'],
                                                                    capsize=capsizi,
                                                                    linestyle=style,
                                                                    linewidth=default_plot_specs['linewidth'])
                else:
                    if line_style_dict[k] == '--':
                        custom_legend_entries_dict[ii] = plt.Line2D((0, 1), (0, 0), color='k', dashes=dashes,
                                                                    linewidth=dash_width_factor*default_plot_specs['linewidth'])
                    else:
                        custom_legend_entries_dict[ii] = plt.Line2D((0, 1), (0, 0), color='k',
                                                                    linestyle=style,
                                                                    linewidth=default_plot_specs['linewidth'])

    if add_dummy_ynames:
        ynames = [ynames_dict[k] for k in range(number_y)]
        # Create custom artists

        simArtist = plt.Line2D((0, 1), (0, 0), color='k', marker='o', linestyle='')
        anyArtist = plt.Line2D((0, 1), (0, 0), color='k')

    #print color_dict

    # print 'printing ynames in funct'
    # print ynames
    #print 'yname dict', ynames_dict

    hl = False
    for jj in range(number_y):
        coli = color_dict[jj]
        style = line_style_dict[jj]  # '--' #'None'
        thickness = default_plot_specs['linewidth']
        if style == '--':
            thickness = thickness*dash_width_factor
            hl = True
            hl_num = 3.6
            dashi = True
        else:
            dashi = False
        if marker:
            if yer is None:
                markstyli = marker_style_dict[jj]
                if ynames is None or jj>len(ynames)-1 or not ynames[jj]:
                    if dashi:
                        ax_1.plot(x[jj], y[jj], color=coli, marker=markstyli
                                  , markersize=default_plot_specs['marker_size'],
                                  dashes=dashes)
                    else:
                        ax_1.plot(x[jj], y[jj], color=coli, marker=markstyli, linestyle=style
                                  , markersize=default_plot_specs['marker_size'],
                                  linewidth=thickness)
                else:
                    if dashi:
                        ax_1.plot(x[jj], y[jj], color=coli, label=ynames[jj], marker=markstyli
                                  , markersize=default_plot_specs['marker_size'],
                                  dashes=dashes)
                    else:
                        ax_1.plot(x[jj], y[jj], color=coli, label=ynames[jj], marker=markstyli,
                                  linestyle=style, markersize=default_plot_specs['marker_size'],
                                  linewidth=thickness)
            # else:
            #     ax_1.plot(x[jj], y[jj], color=coli,linestyle=style)
        else:
            if ynames is None or jj > len(ynames) - 1:
                if dashi:
                    ax_1.plot(x[jj], y[jj], color=coli, linewidth=thickness,dashes=dashes)
                else:
                    ax_1.plot(x[jj], y[jj], color=coli, linewidth=thickness, linestyle=style)
            else:
                if dashi:
                    ax_1.plot(x[jj], y[jj], color=coli, linewidth=thickness,label=ynames[jj],dashes=dashes)
                else:
                    ax_1.plot(x[jj], y[jj], color=coli, linewidth=thickness, linestyle=style,
                              label=ynames[jj])



        if yer is not None:

            # ax_1.plot(x[jj], yer_datas_high, color=coli,
            #               label=y_names[jj] + ' + SE', linestyle='--')
            # ax_1.plot(x[jj], yer_datas_low, color=coli,
            #               label=y_names[jj] + ' - SE', linestyle='--')
            if marker:
                markstyli = marker_style_dict[jj]
                if markstyli and not style:
                    capsizi = default_plot_specs['cap_size']
                else:
                    capsizi = None
                if ynames is None or jj > len(ynames) - 1:
                    if dashi:
                        ax_1.errorbar(x[jj],y[jj], yer[jj], color=coli,marker=markstyli,
                                      markersize=default_plot_specs['marker_size'],
                                      capsize=capsizi,
                                      linewidth=default_plot_specs['linewidth'],dashes=dashes)
                    else:
                        ax_1.errorbar(x[jj],y[jj], yer[jj], color=coli,marker=markstyli,
                                      markersize=default_plot_specs['marker_size'],
                                      capsize=capsizi,
                                      linewidth=default_plot_specs['linewidth'],linestyle=style)
                else:
                    if dashi:
                        ax_1.errorbar(x[jj],y[jj], yer[jj], color=coli,marker=markstyli,
                                      markersize=default_plot_specs['marker_size'],
                                      capsize=capsizi,
                                      label=y_names[jj],
                                      linewidth=default_plot_specs['linewidth'],dashes=dashes)
                    else:
                        ax_1.errorbar(x[jj],y[jj], yer[jj], color=coli,marker=markstyli,
                                      markersize=default_plot_specs['marker_size'],
                                      capsize=capsizi,
                                      label=y_names[jj],
                                      linewidth=default_plot_specs['linewidth'],linestyle=style)
            else:
                yer_datas_high = [y_i + y_er_i for y_i, y_er_i in zip(y[jj], yer[jj])]
                yer_datas_low = [y_i - y_er_i for y_i, y_er_i in zip(y[jj], yer[jj])]
                ax_1.plot(x[jj], yer_datas_high, color=coli, linestyle='--',dashes=dashes)
                ax_1.plot(x[jj], yer_datas_low, color=coli, linestyle='--',dashes=dashes)

    if default_plot_specs['yrotation'] is 'vertical':
        if default_plot_specs['ylabelspace'] ==0:
            ax_1.set_ylabel(y_label, **default_plot_specs['axis_font'])
        else:
            labpad = int(default_plot_specs['axis_font']['size'])*default_plot_specs['ylabelspace']
            ax_1.set_ylabel(y_label,labelpad=labpad, **default_plot_specs['axis_font'])
    else:
        labpad =int(default_plot_specs['axis_font']['size'])*3
        #ax_1.set_ylabel(y_label,rotation=plotspecs['yrotation'],labelpad=int(labpad), **default_plot_specs['axis_font'])
        ax_1.set_ylabel(y_label, rotation=default_plot_specs['yrotation'],labelpad=labpad, horizontalalignment = 'center',verticalalignment ='center',
                        **default_plot_specs['axis_font'])


    # Set the tick labels font
    for labeli in (ax_1.get_xticklabels() + ax_1.get_yticklabels()):
        # labeli.set_fontname('Arial')
        labeli.set_fontsize(default_plot_specs['ticksize'])

    ax_1.set_xlabel(x_label, **default_plot_specs['axis_font'])


    xlow, xhigh = min(x[0]), max(x[0])
    for xx in x[1:]:
        mycopy_low = [g for g in copy.deepcopy(xx)]
        mycopy_high = [g for g in copy.deepcopy(xx)]
        mycopy_low.append(xlow)
        mycopy_high.append(xhigh)
        xlow, xhigh = min(mycopy_low), max(mycopy_high)
    # set axes limits
    if domain is None:
        extra = (xhigh-xlow)*default_plot_specs['x_scale']
        xlow -= extra
        xhigh +=extra


    #Make vertical lines
    for xfloat in vlines:
        if xlow < xfloat < xhigh:
            ax_1.axvline(x=xfloat,color = default_plot_specs['vlinecolor'],linestyle= default_plot_specs['vlinestyle'],linewidth=default_plot_specs['vlineswidth'])

    # if not marker:
    #     xhigh -= 15

    if yrange is None:
        if y:
            if y[0]:
                if yer is not None:
                    ylow, yhigh = min([yi-yi_er for yi, yi_er in zip(y[0],yer[0])]), max([yi+yi_er for yi, yi_er in zip(y[0],yer[0])])
                else:
                    ylow, yhigh = min(y[0]), max(y[0])
            else:
                ylow, yhigh = 0, 0
        else:
            ylow, yhigh = 0, 0
        if yer is not None:
            for yy, yy_er in zip(y[1:],yer[1:]):
                ylow, yhigh = min([ylow] + [yi-yi_er for yi, yi_er in zip(yy,yy_er)]), max([yhigh]+ [yi+yi_er for yi, yi_er in zip(yy,yy_er)])
        else:
            for yy in y[1:]:
                 ylow, yhigh = min([ylow] + yy), max([yhigh] + yy)
        extra = (yhigh-ylow)*default_plot_specs['y_scale']
        ylow -= extra
        yhigh +=extra


    ax_1.set_xlim(xlow, xhigh)
    ax_1.set_ylim(ylow, yhigh)

    while under_text:
        texti = under_text.pop(0)
        plt.figtext(0.08, text_heights.pop(0), texti, default_plot_specs['undertext_font'])

    if text:
        ax_1.text(default_plot_specs['text_loc'][0], default_plot_specs['text_loc'][1], text,
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax_1.transAxes,
                color=default_plot_specs['text_color'], fontsize=default_plot_specs['text_size'])

    #print 'display_leg_numbers', display_leg_numbers


    if default_plot_specs['xshade']:
        ax_1.axvspan(default_plot_specs['xshade'][0], default_plot_specs['xshade'][1], alpha=0.3, color=default_plot_specs['xshade_color'])

    if ynames:
        # print 'the display leg numbers '
        # print display_leg_numbers

        handles, labels = ax_1.get_legend_handles_labels()
        handles = [handle for i,handle in enumerate(handles) if i in display_leg_numbers]
        labels = [label for i,label in enumerate(labels) if i in display_leg_numbers]
        if groupings_labels_within:
            mini = min(len(list(custom_legend_entries_dict.keys())),len(groupings_labels_within))
            handles += [custom_legend_entries_dict[k] for k in range(mini)]
            labels += groupings_labels_within[:mini]
        if hl:
            lgd = ax_1.legend(handles, labels, loc=default_plot_specs['legend_anchor'],
                              bbox_to_anchor=default_plot_specs['legend_loc'],
                              prop=default_plot_specs['legend_font'], ncol=n_legend_columns,handlelength=hl_num)
        else:
            lgd = ax_1.legend(handles, labels, loc=default_plot_specs['legend_anchor'],
                              bbox_to_anchor=default_plot_specs['legend_loc'],
                              prop=default_plot_specs['legend_font'], ncol=n_legend_columns)

        if legend_title:
            lgd.set_title(legend_title,prop=default_plot_specs['legend_font'])

        plt.setp(lgd.get_title(), multialignment='center')

        # if hl:
        #     print 'doing hl 2'
        #     ax_1.legend(handlelength=2)


    if default_plot_specs['nxticks'] > 0:
        #visible_labelsx = [lab for lab in ax_1.get_xticklabels() if lab.get_visible() is True and lab.get_text() != '']
        for lab in ax_1.get_xticklabels():
            lab.set_visible(True)
        visible_labelsx = [lab for lab in ax_1.get_xticklabels() if lab.get_visible() is True]
        visible_labelsx=visible_labelsx[1::default_plot_specs['nxticks']]
        plt.setp(visible_labelsx, visible = False)
        #
        #ax_1.set_xticks(visible_labelsx[1::2])
        #plt.setp(visible_labels[1::2], visible=False)
        #ax_1.locator_params(axis='x', nticks=default_plot_specs['nxticks'])
    #
    if default_plot_specs['nyticks'] > 0:
    #     #ax_1.locator_params(axis='y', nticks=default_plot_specs['nyticks'])
        visible_labelsy = [lab for lab in ax_1.get_yticklabels() if lab.get_visible() is True]
        if len(visible_labelsy) > 4:
            visible_labelsy = visible_labelsy[2:-2]
            plt.setp(visible_labelsy, visible=False)

    #plt.grid('off')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir,'%s.png' % the_label)

    if save:
        save_fig(fig, save_dir)
    else:
        return fig, save_dir


def add_y_axis(fig, x, y, yer = None, ylabel = None, ynames = None, domain = None,
               yrange = None, marker = None, markerstyles=None, plotspecs = None, groupings = None, legend_title = None, n_legend_columns = None,
               linestyles = None, colors = None, axiscolour= None):


    default_plot_specs = copy.deepcopy(default_plot_specs_all)
    default_plot_specs['legend_font'] = {'size': 8}
    default_plot_specs['legend_anchor'] = 'upper right'
    default_plot_specs['legend_loc'] = (0.98, -0.1)

    if marker:
        default_plot_specs['x_scale'] = 0.05
    else:
        default_plot_specs['x_scale'] = 0

    if plotspecs is not None:
        for stat in list(default_plot_specs.keys()):
            if stat in plotspecs:
                default_plot_specs[stat] = plotspecs[stat]


    axes = fig.get_axes()
    axes.append(axes[0].twinx())

    if axiscolour is None:
        axiscolour = 'k'

    if ylabel is not None:
        axes[1].set_ylabel(ylabel, color=axiscolour, **default_plot_specs['axis_font'])
        if axiscolour is not None:
            axes[1].spines['right'].set_color(axiscolour)
            axes[1].tick_params(axis='y', colors=axiscolour)

    # Set the tick labels font
    for jj in range(len(axes)):
        for labeli in (axes[jj].get_xticklabels() + axes[jj].get_yticklabels()):
            # labeli.set_fontname('Arial')
            labeli.set_fontsize(default_plot_specs['ticksize'])


    if marker is None:
        marker = False
    if n_legend_columns is None:
        n_legend_columns = 1

    number_y = len(y)


    if groupings is None:
        grouped = False
        groupings = [{i} for i in range(number_y)]
    else:
        grouped = True

    # Make sure all the elements are in a colour grouping
    if grouped:
        extra_group = set()
        for i in range(number_y):
            in_a_group = False
            for seti in groupings:
                for el in seti:
                    if i == el:
                        if not in_a_group:
                            in_a_group = True
                        else:
                            print(el, ' in two colour groups')
            if not in_a_group:
                extra_group.add(i)



    if domain is not None:
        xlow = domain[0]
        xhigh = domain[1]
        for ii in range(number_y):
            klow = x[ii].index(find_nearest(x[ii],xlow))
            khigh = x[ii].index(find_nearest(x[ii], xhigh))
            x[ii] = x[ii][klow:khigh]
            y[ii] = y[ii][klow:khigh]
    if yrange is not None:
        ylow = yrange[0]
        yhigh = yrange[1]

    if ylabel is None:
        y_label = ''
        the_label = 'y_' +str(number_y) +'_'
    else:
        y_label = ylabel
    if ynames is None:
        y_names = []
    else:
        y_names = ynames


    if marker:
        rcParams['legend.numpoints'] = 1


    #mycolors = cm.gnuplot2(np.linspace(0, 1, len(groupings)))
    mycolors = cm.rainbow(np.linspace(0, 1, len(groupings)))

    color_dict = dict()
    line_style_dict = dict()
    marker_style_dict = dict()

    for seti, jj in zip(groupings, range(number_y)):
        for k,ii in zip(sorted(list(seti)), range(len(seti))):
            if colors is None:
                color_dict[k] = mycolors[jj]
                if number_y == 1:
                    color_dict[k] = axiscolour
            else:
                color_dict[k] = colors[jj]
            if grouped:
                marker_style_dict[k] = marker_styles[ii]
                line_style_dict[k] = line_styles[ii]

            else:
                if markerstyles is None:
                    marker_style_dict[k] = default_plot_specs['marker_style']
                else:
                    marker_style_dict[k] = markerstyles[k]
                if linestyles is None:
                    line_style_dict[k] = default_plot_specs['linestyle']
                else:
                    line_style_dict[k] = linestyles[k]

    hl = False
    for jj in range(number_y):
        coli = color_dict[jj]
        style = line_style_dict[jj]  # '--' #'None'
        thickness = default_plot_specs['linewidth']
        if style == '--':
            dashes = (20, 45)
            thickness = thickness*2
            hl = True
        else:
            dashes = False
        if marker:
            if yer is None:
                markerstyli = marker_style_dict[jj]
                if ynames is None or jj>len(ynames)-1:
                    if dashes:
                        axes[1].plot(x[jj], y[jj], color=coli, marker=markerstyli,
                                  markersize=default_plot_specs['marker_size'],
                                     dashes=dashes)
                    else:
                        axes[1].plot(x[jj], y[jj], color=coli, marker=markerstyli, linestyle=style
                                  , markersize=default_plot_specs['marker_size'])
                else:
                    if dashes:
                        axes[1].plot(x[jj], y[jj], color=coli, label=ynames[jj], marker=markerstyli
                                  , markersize=default_plot_specs['marker_size'],
                                     dashes=dashes)
                    else:
                        axes[1].plot(x[jj], y[jj], color=coli, label=ynames[jj], marker=markerstyli,
                                  linestyle=style, markersize=default_plot_specs['marker_size'])
            # else:
            #     ax_1.plot(x[jj], y[jj], color=coli,linestyle=style)
        else:
            if ynames is None or jj > len(ynames) - 1:
                if dashes:
                    axes[1].plot(x[jj], y[jj], color=coli, linewidth=thickness,dashes=dashes)
                else:
                    axes[1].plot(x[jj], y[jj], color=coli, linewidth=thickness, linestyle=style)
            else:
                if dashes:
                    axes[1].plot(x[jj], y[jj], color=coli, linewidth=thickness,label=ynames[jj],dashes=dashes)
                else:
                    axes[1].plot(x[jj], y[jj], color=coli, linewidth=thickness, linestyle=style,
                              label=ynames[jj])


        if yer is not None:

            # ax_1.plot(x[jj], yer_datas_high, color=coli,
            #               label=y_names[jj] + ' + SE', linestyle='--')
            # ax_1.plot(x[jj], yer_datas_low, color=coli,
            #               label=y_names[jj] + ' - SE', linestyle='--')
            if marker:
                markstyli = marker_style_dict[jj]
                if markstyli:
                    capsizi = default_plot_specs['cap_size']
                else:
                    capsizi = None
                if ynames is None or jj > len(ynames) - 1:

                    if dashes:
                        axes[1].errorbar(x[jj],y[jj], yer[jj], color=coli,marker=markstyli,
                                      markersize=default_plot_specs['marker_size'],
                                         capsize=capsizi,
                                      linewidth=default_plot_specs['linewidth'],dashes=dashes)
                    else:
                        axes[1].errorbar(x[jj],y[jj], yer[jj], color=coli,marker=markstyli,
                                      markersize=default_plot_specs['marker_size'],
                                         capsize=capsizi,
                                      linewidth=default_plot_specs['linewidth'],linestyle=style)
                else:
                    if dashes:
                        axes[1].errorbar(x[jj],y[jj], yer[jj], color=coli,marker=markstyli,
                                      markersize=default_plot_specs['marker_size'],
                                         capsize=capsizi, label=y_names[jj],
                                      linewidth=default_plot_specs['linewidth'],dashes=dashes)
                    else:
                        axes[1].errorbar(x[jj],y[jj], yer[jj], color=coli,marker=markstyli,
                                      markersize=default_plot_specs['marker_size'],
                                         capsize=capsizi,label=y_names[jj],
                                      linewidth=default_plot_specs['linewidth'],linestyle=style)

            else:
                yer_datas_high = [y_i + y_er_i for y_i, y_er_i in zip(y[jj], yer[jj])]
                yer_datas_low = [y_i - y_er_i for y_i, y_er_i in zip(y[jj], yer[jj])]
                axes[1].plot(x[jj], yer_datas_high, color=coli, linestyle='--',dashes=(5,20))
                axes[1].plot(x[jj], yer_datas_low, color=coli, linestyle='--',dashes=(5,20))

    if default_plot_specs['yrotation'] is 'vertical':
        axes[1].set_ylabel(y_label, **default_plot_specs['axis_font'])
    else:
        labpad =int(default_plot_specs['axis_font']['size'])*3
        #ax_1.set_ylabel(y_label,rotation=plotspecs['yrotation'],labelpad=int(labpad), **default_plot_specs['axis_font'])
        axes[1].set_ylabel(y_label, rotation=default_plot_specs['yrotation'],labelpad=labpad, horizontalalignment = 'center',verticalalignment ='center',
                        **default_plot_specs['axis_font'])


    # Set the tick labels font
    for labeli in (axes[1].get_xticklabels() + axes[1].get_yticklabels()):
        # labeli.set_fontname('Arial')
        labeli.set_fontsize(default_plot_specs['ticksize'])

    #ax_1.set_xlabel(x_label, **default_plot_specs['axis_font'])

    if hl:
        plt.legend(handlelength=250)


    # set axes limits
    if domain is None:
        xlow, xhigh = min(x[0]), max(x[0])
        for xx in x[1:]:
            mycopy_low = [g for g in copy.deepcopy(xx)]
            mycopy_high = [g for g in copy.deepcopy(xx)]
            mycopy_low.append(xlow)
            mycopy_high.append(xhigh)
            xlow, xhigh = min(mycopy_low), max(mycopy_high)
        extra = (xhigh-xlow)*default_plot_specs['x_scale']
        xlow -= extra
        xhigh +=extra


    # if not marker:
    #     xhigh -= 15

    if yrange is None:
        if y:
            if y[0]:
                ylow, yhigh = min(y[0]), max(y[0])
            else:
                ylow, yhigh = 0, 0
        else:
            ylow, yhigh = 0, 0
        for yy in y[1:]:
            ylow, yhigh = min([ylow] + yy), max([yhigh]+ yy)
        extra = (yhigh-ylow)*default_plot_specs['y_scale']
        ylow -= extra
        yhigh +=extra

    axes[1].set_xlim(xlow, xhigh)
    axes[1].set_ylim(ylow, yhigh)


    if ynames:
        handles, labels = axes[1].get_legend_handles_labels()
        lgd = axes[1].legend(handles, labels, loc=default_plot_specs['legend_anchor'],
                          bbox_to_anchor=default_plot_specs['legend_loc'],
                          prop = default_plot_specs['legend_font'],ncol=n_legend_columns)
        if legend_title:
            lgd.set_title(legend_title,prop=default_plot_specs['legend_font'])

        plt.setp(lgd.get_title(), multialignment='center')

    #
    if default_plot_specs['nyticks'] > 0:
    #     #ax_1.locator_params(axis='y', nticks=default_plot_specs['nyticks'])
        visible_labelsy = [lab for lab in axes[1].get_yticklabels() if lab.get_visible() is True]
        if len(visible_labelsy) > 4:
            visible_labelsy = visible_labelsy[2:-2]
            plt.setp(visible_labelsy, visible=False)

    return fig



def plot_many_y_SMBE(x, y, yer=None, xlabel = None, ylabel = None, ynames = None, label = None, domain=None,
                     yrange = None, undertext =None, savedir = None, marker=None, plotspecs = None, groupings=None,
                     vlines = None, legend_title=None, n_legend_columns=None, text=None):
    """
    Plots multiple y against x, on a single set of axes. Saves the plot.

    Parameters:
        x: (list) A list [x_1,x_2,...] with every x_i a list of x values
        y: (list) A list [y_1, y_2...] with every y_i a list of y values
        yer: (list) A list [yer_1, yer_2,...] with every yer_i a list of the standard error on y_i
        marker: (boolean) If true, use markers instead of line
        xlabel: (string) The name of the x-axis
        ylabel: (float) The name of the y-axis
        ynames: (list) A list of the names of the things being plotted
        legend_title: (string) A label for the legend
        n_legend_columns: (int) Number of columns in the legend
        label: (string) A label for the plot
        domain: (list) Form [xlow, xhigh] - the domain of the plot
        yrange: (list) Form [ylow, yhigh] - the yrange of the plot
        undertext: (list) List of strings to be placed beneath the plot
        text: (string) A string to be printed on the plot
        savedir: (string) The name of the directory to save the graph in
        groupings: (list of sets) each set has the index of plots to be the same colour
        vlines: (list of floats or float) Values at which a vertical line is to be plotted
        plotspecs: (dict) A dictionary to set the plot specifications if they differ from default -
            Possible keys:
                fsize: (tup) Gives dimensions of the plot (e.g. (11,4))
                linewidth: (float) Width of the lines in the plot
                axis_font: (dict) (e.g. {'fontname': 'Arial', 'size': '18'})
                ticksize: (float) Size of the ticks
                nxticks: (int) Number of xticks
                dpi: (float) Quality of the image
                undertext_font: (dict) Font of undertext
                    (e.g. {'color': 'black', 'weight': 'roman', 'size': 'x-small'})
                text_color: (string) Colour of the text on the plot (e.g. 'green')
                text_loc: (list) [x_pos, y_pos] of the text (e.g. [0.8,0.5])
                text_size: (float) size of the text
                x_scale: (float) The x-axis is (1+2x_scale)*(xmax -xmin) wide, if domain is None
                y_scale: (float) The y-axis is (1+2y_scale)*(ymax -ymin) high, if domain is None
                legend_font: (dict) Size and font of legend writing (e.g. {'size': 8})
                legend_anchor: (string) The point on the legend that's anchored (e.g. upper right)
                legend_loc: (tup) Point on plot where the legend is anchored (e.g. (0.98, -0.1))
                vlinestyle: (string) The style of vertical line
                vlinecolor: (string) The colour of vertical lines
                vlineswidth: (float) Width of the vline

    """
    if savedir is None:
        save_dir = os.getcwd()
    else:
        save_dir = savedir
    if marker is None:
        marker = False
    if vlines is None:
        vlines = []
    if isinstance(vlines, float):
        vlines = [vlines]
    if n_legend_columns is None:
        n_legend_columns = 1

    number_y = len(y)

    if groupings is None:
        grouped = False
        groupings = [{i} for i in range(number_y)]
    else:
        grouped = True

    # Make sure all the elements are in a colour grouping
    if grouped:
        extra_group = set()
        for i in range(number_y):
            in_a_group = False
            for seti in groupings:
                for el in seti:
                    if i == el:
                        if not in_a_group:
                            in_a_group = True
                        else:
                            print(el, ' in two colour groups')
            if not in_a_group:
                extra_group.add(i)


    default_plot_specs = copy.deepcopy(default_plot_specs_all)
    default_plot_specs['legend_font'] = {'size': 8}
    default_plot_specs['legend_anchor'] = 'upper right'
    default_plot_specs['legend_loc'] = (0.98, -0.1)

    if marker:
        default_plot_specs['x_scale'] = 0.05
    else:
        default_plot_specs['x_scale'] = 0

    text_heights = [-0.023, -0.069, -0.115,-0.161]

    if plotspecs is not None:
        for stat in list(default_plot_specs.keys()):
            if stat in plotspecs:
                default_plot_specs[stat] = plotspecs[stat]

    the_label = ''

    if domain is not None:
        xlow = domain[0]
        xhigh = domain[1]
        for ii in range(number_y):
            klow = x[ii].index(find_nearest(x[ii],xlow))
            khigh = x[ii].index(find_nearest(x[ii], xhigh))
            x[ii] = x[ii][klow:khigh]
            y[ii] = y[ii][klow:khigh]
    if yrange is not None:
        ylow = yrange[0]
        yhigh = yrange[1]
    if xlabel is None:
        x_label = ''
    else:
        x_label = xlabel
    if ylabel is None:
        y_label = ''
        the_label = 'y_' +str(number_y) +'_'
    else:
        y_label = ylabel
        the_label += y_label[:4] +'_'
    if ynames is None:
        y_names = []
    else:
        y_names = ynames
    if label is None:
        the_label = the_label + 'vs_' +x_label
    else:
        the_label = label

    under_text = []
    if undertext is not None:
        under_text = undertext[:]

    if marker:
        rcParams['legend.numpoints'] = 1

    plt.clf()

    fig = plt.figure(figsize=default_plot_specs['fsize'], dpi=default_plot_specs['dpi'])
    ax_1 = fig.add_subplot(111)

    colors = cm.rainbow(np.linspace(0, 1, len(groupings)))
    color_dict = dict()
    line_style_dict = dict()
    marker_style_dict = dict()

    for seti, jj in zip(groupings, range(number_y)):
        for k,ii in zip(sorted(list(seti)), range(len(seti))):
            color_dict[k] = colors[jj]
            if grouped:
                marker_style_dict[k] = marker_styles[ii]
                line_style_dict[k] = line_styles[ii]

            else:
                marker_style_dict[k] = default_plot_specs['marker_style']
                line_style_dict[k] = default_plot_specs['linestyle']


    for jj in range(number_y):
        coli = color_dict[jj]

        if marker:
            style = line_style_dict[jj]#'--' #'None'
            if yer is None:
                if ynames is None or jj>len(ynames)-1:
                    ax_1.plot(x[jj], y[jj], color=coli, marker=marker_style_dict[jj], linestyle=style
                              , markersize=default_plot_specs['marker_size'])
                else:
                    ax_1.plot(x[jj], y[jj], color=coli, label=ynames[jj], marker=marker_style_dict[jj],linestyle=style
                              , markersize=default_plot_specs['marker_size'])
            # else:
            #     ax_1.plot(x[jj], y[jj], color=coli,linestyle=style)
        else:
            style = line_style_dict[jj]
            if ynames is None or jj > len(ynames) - 1:
                ax_1.plot(x[jj], y[jj], color=coli, linewidth=default_plot_specs['linewidth'],linestyle=style)
            else:
                ax_1.plot(x[jj], y[jj], color=coli, linewidth=default_plot_specs['linewidth'],linestyle=style,label=ynames[jj])


        if yer is not None:

            # ax_1.plot(x[jj], yer_datas_high, color=coli,
            #               label=y_names[jj] + ' + SE', linestyle='--')
            # ax_1.plot(x[jj], yer_datas_low, color=coli,
            #               label=y_names[jj] + ' - SE', linestyle='--')
            if marker:
                markersyli = default_plot_specs['marker_size']
                if markersyli and not style:
                    capsizi = default_plot_specs['cap_size']
                else:
                    capsizi = None
                ax_1.errorbar(x[jj],y[jj], yer[jj], color=coli,marker=markersyli,
                              markersize=default_plot_specs['marker_size'],capsize=capsizi,
                              label=y_names[jj],
                              linewidth=default_plot_specs['linewidth'],linestyle=style)
            else:
                yer_datas_high = [y_i + y_er_i for y_i, y_er_i in zip(y[jj], yer[jj])]
                yer_datas_low = [y_i - y_er_i for y_i, y_er_i in zip(y[jj], yer[jj])]
                ax_1.plot(x[jj], yer_datas_high, color=coli, linestyle='--')
                ax_1.plot(x[jj], yer_datas_low, color=coli, linestyle='--')

    ax_1.set_ylabel(y_label, **default_plot_specs['axis_font'])

    if default_plot_specs['nxticks'] > 0:
        ax_1.locator_params(axis='x', nticks=default_plot_specs['nxticks'])

    if default_plot_specs['nyticks'] > 0:
        ax_1.locator_params(axis='y', nticks=default_plot_specs['nyticks'])


    # Set the tick labels font
    for labeli in (ax_1.get_xticklabels() + ax_1.get_yticklabels()):
        # labeli.set_fontname('Arial')
        labeli.set_fontsize(default_plot_specs['ticksize'])

    ax_1.set_xlabel(x_label, **default_plot_specs['axis_font'])


    # set axes limits
    if domain is None:
        xlow, xhigh = min(x[0]), max(x[0])
        for xx in x[1:]:
            mycopy_low = [g for g in copy.deepcopy(xx)]
            mycopy_high = [g for g in copy.deepcopy(xx)]
            mycopy_low.append(xlow)
            mycopy_high.append(xhigh)
            xlow, xhigh = min(mycopy_low), max(mycopy_high)
        extra = (xhigh-xlow)*default_plot_specs['x_scale']
        xlow -= extra
        xhigh +=extra


    #Make vertical lines
    for xfloat in vlines:
        if xlow < xfloat < xhigh:
            ax_1.axvline(x=xfloat,color = default_plot_specs['vlinecolor'],linestyle= default_plot_specs['vlinestyle'],linewidth=default_plot_specs['vlineswidth'])

    if not marker:
        xhigh -= 15

    if yrange is None:
        if y:
            if y[0]:
                ylow, yhigh = min(y[0]), max(y[0])
            else:
                ylow, yhigh = 0, 0
        else:
            ylow, yhigh = 0, 0
        for yy in y[1:]:
            ylow, yhigh = min([ylow] + yy), max([yhigh]+ yy)
        extra = (yhigh-ylow)*default_plot_specs['y_scale']
        ylow -= extra
        yhigh +=extra



    ax_1.set_xlim(xlow, xhigh)
    ax_1.set_ylim(ylow, yhigh)

    while under_text:
        texti = under_text.pop(0)
        plt.figtext(0.08, text_heights.pop(0), texti, default_plot_specs['undertext_font'])

    if text:
        ax_1.text(default_plot_specs['text_loc'][0], default_plot_specs['text_loc'][1], text,
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax_1.transAxes,
                color=default_plot_specs['text_color'], fontsize=default_plot_specs['text_size'])

    if ynames:
        handles, labels = ax_1.get_legend_handles_labels()
        lgd = ax_1.legend(handles, labels, loc=default_plot_specs['legend_anchor'],
                          bbox_to_anchor=default_plot_specs['legend_loc'],
                          prop = default_plot_specs['legend_font'],ncol=n_legend_columns)
        if legend_title:
            lgd.set_title(legend_title,prop=default_plot_specs['legend_font'])

    plt.grid('off')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir,'%s.png' % the_label)

    if ynames:
        plt.savefig(save_dir, bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        plt.savefig(save_dir, bbox_inches='tight')  # ,bbox_extra_artists=(lgd,))

    plt.close('all')



def plot_many_y_break_x(x, y, yer=None, xlabel = None, ylabel = None, ynames = None, label = None, domain=None, domain_2=None, frac=None,
                        yrange = None, undertext =None, savedir = None, marker=None, plotspecs = None, groupings=None,
                        vlines = None, legend_title=None, n_legend_columns=None, text=None):
    """
    Plots multiple y against x, on a single set of axes. Saves the plot.

    Parameters:
        x: (list) A list [x_1,x_2,...] with every x_i a list of x values
        y: (list) A list [y_1, y_2...] with every y_i a list of y values
        yer: (list) A list [yer_1, yer_2,...] with every yer_i a list of the standard error on y_i
        marker: (boolean) If true, use markers instead of line
        xlabel: (string) The name of the x-axis
        ylabel: (float) The name of the y-axis
        ynames: (list) A list of the names of the things being plotted
        legend_title: (string) A label for the legend
        n_legend_columns: (int) Number of columns in the legend
        label: (string) A label for the plot
        domain: (list) Form [xlow, xhigh] - the domain of the plot
        domain_2: (list) Form [xlow,xhigh] - the domain after the break in the x-axis
        frac: (list) e.g. [0.5,0.5] fraction of plot taken up by each part of the x-axis
        yrange: (list) Form [ylow, yhigh] - the yrange of the plot
        undertext: (list) List of strings to be placed beneath the plot
        text: (string) A string to be printed on the plot
        savedir: (string) The name of the directory to save the graph in
        groupings: (list of sets) each set has the index of plots to be the same colour
        vlines: (list of floats or float) Values at which a vertical line is to be plotted
        plotspecs: (dict) A dictionary to set the plot specifications if they differ from default -
            Possible keys:
                fsize: (tup) Gives dimensions of the plot (e.g. (11,4))
                linewidth: (float) Width of the lines in the plot
                axis_font: (dict) (e.g. {'fontname': 'Arial', 'size': '18'})
                ticksize: (float) Size of the ticks
                nxticks: (int) Number of xticks
                dpi: (float) Quality of the image
                undertext_font: (dict) Font of undertext
                    (e.g. {'color': 'black', 'weight': 'roman', 'size': 'x-small'})
                text_color: (string) Colour of the text on the plot (e.g. 'green')
                text_loc: (list) [x_pos, y_pos] of the text (e.g. [0.8,0.5])
                text_size: (float) size of the text
                x_scale: (float) The x-axis is (1+2x_scale)*(xmax -xmin) wide, if domain is None
                y_scale: (float) The y-axis is (1+2y_scale)*(ymax -ymin) high, if domain is None
                legend_font: (dict) Size and font of legend writing (e.g. {'size': 8})
                legend_anchor: (string) The point on the legend that's anchored (e.g. upper right)
                legend_loc: (tup) Point on plot where the legend is anchored (e.g. (0.98, -0.1))
                vlinestyle: (string) The style of vertical line
                vlinecolor: (string) The colour of vertical lines
                vlineswidth: (float) Width of the vline

    """
    if savedir is None:
        save_dir = os.getcwd()
    else:
        save_dir = savedir
    if marker is None:
        marker = False
    if vlines is None:
        vlines = []
    if isinstance(vlines, float):
        vlines = [vlines]
    if n_legend_columns is None:
        n_legend_columns = 1
    if domain_2 is None:
        split = False
    else:
        split = True
        if frac is None:
            frac = [0.5,0.5]

    number_y = len(y)

    if groupings is None:
        grouped = False
        groupings = [{i} for i in range(number_y)]
    else:
        grouped = True

    # Make sure all the elements are in a colour grouping
    if grouped:
        extra_group = set()
        for i in range(number_y):
            in_a_group = False
            for seti in groupings:
                for el in seti:
                    if i == el:
                        if not in_a_group:
                            in_a_group = True
                        else:
                            print(el, ' in two colour groups')
            if not in_a_group:
                extra_group.add(i)


    default_plot_specs = copy.deepcopy(default_plot_specs_all)
    default_plot_specs['legend_font'] = {'size': 8}
    default_plot_specs['legend_anchor'] = 'upper right'
    default_plot_specs['legend_loc'] = (0.98, -0.1)

    if marker:
        default_plot_specs['x_scale'] = 0.05
    else:
        default_plot_specs['x_scale'] = 0

    text_heights = [-0.023, -0.069, -0.115,-0.161]

    if plotspecs is not None:
        for stat in list(default_plot_specs.keys()):
            if stat in plotspecs:
                default_plot_specs[stat] = plotspecs[stat]

    the_label = ''

    if domain is not None:
        xlow1 = domain[0]
        xhigh1 = domain[1]

    if domain_2 is not None:
        xlow2 = domain_2[0]
        xhigh2 = domain_2[1]

    if yrange is not None:
        ylow = yrange[0]
        yhigh = yrange[1]
    if xlabel is None:
        x_label = ''
    else:
        x_label = xlabel
    if ylabel is None:
        y_label = ''
        the_label = 'y_' +str(number_y) +'_'
    else:
        y_label = ylabel
        the_label += y_label[:4] +'_'
    if ynames is None:
        y_names = []
    else:
        y_names = ynames
    if label is None:
        the_label = the_label + 'vs_' +x_label
    else:
        the_label = label

    under_text = []
    if undertext is not None:
        under_text = undertext[:]

    if marker:
        rcParams['legend.numpoints'] = 1

    plt.clf()


    Naxes = 1
    if split:
        Naxes = 2
        fig,(ax_1, ax_2) =plt.subplots(1, 2,sharey=True, gridspec_kw={'width_ratios': [1, 3]},figsize=default_plot_specs['fsize'], dpi=default_plot_specs['dpi'])

        # # Margins (dimensions are in inches):
        # left_margin = 0.6 / fig.get_figwidth()
        # right_margin = 0.25 / fig.get_figwidth()
        # bottom_margin = 0.75 / fig.get_figheight()
        # top_margin = 0.25 / fig.get_figwidth()
        # mid_margin = 0.1 / fig.get_figwidth()  # horizontal space between subplots
        # x0, y0 = left_margin, bottom_margin  # origin point of the axe
        # h = 1 - (bottom_margin + top_margin)  # height of the axe
        #
        # # total width of the axes:
        # wtot = 1 - (left_margin + right_margin + (Naxes - 1) * mid_margin)
        #
        # w1 = wtot * frac[0]
        # w2 = wtot * frac[1]

        #ax_1 = fig.add_axes([x0, y0, w1, h], frameon=True, axisbg='none')
        ax_1.spines['right'].set_visible(False)
        ax_1.tick_params(right='off', labelright='off')

        #x0 += w1 + mid_margin
        #ax_2 = fig.add_axes([x0, y0, w2, h], frameon=True, axisbg='none')

        ax_2.spines['left'].set_visible(False)
        ax_2.tick_params(left='off', labelleft='off',
                       right='off', labelright='off')
        axes = [ax_1,ax_2]
    else:
        fig = plt.figure(figsize=default_plot_specs['fsize'], dpi=default_plot_specs['dpi'])
        ax_1 = fig.add_subplot(111)
        axes = [ax_1]

    colors = cm.rainbow(np.linspace(0, 1, len(groupings)))
    color_dict = dict()
    line_style_dict = dict()
    marker_style_dict = dict()

    for seti, jj in zip(groupings, range(number_y)):
        for k,ii in zip(sorted(list(seti)), range(len(seti))):
            color_dict[k] = colors[jj]
            if grouped:
                marker_style_dict[k] = marker_styles[ii]
                line_style_dict[k] = line_styles[ii]

            else:
                marker_style_dict[k] = default_plot_specs['marker_style']
                line_style_dict[k] = default_plot_specs['linestyle']



    ax_num =0
    for ax in axes:
        for jj in range(number_y):
            coli = color_dict[jj]

            if marker:
                style = line_style_dict[jj]#'--' #'None'
                if yer is None:
                    if ynames is None or jj>len(ynames)-1 or ax_num >0:
                        ax.plot(x[jj], y[jj], color=coli, marker=marker_style_dict[jj], linestyle=style
                                  , markersize=default_plot_specs['marker_size'])
                    else:
                        ax.plot(x[jj], y[jj], color=coli, label=ynames[jj], marker=marker_style_dict[jj],linestyle=style
                                  , markersize=default_plot_specs['marker_size'])
                # else:
                #     ax_1.plot(x[jj], y[jj], color=coli,linestyle=style)
            else:
                style = line_style_dict[jj]
                if ynames is None or jj > len(ynames) - 1 or ax_num>0:
                    ax.plot(x[jj], y[jj], color=coli, linewidth=default_plot_specs['linewidth'],linestyle=style)
                else:
                    ax.plot(x[jj], y[jj], color=coli, linewidth=default_plot_specs['linewidth'],linestyle=style,label=ynames[jj])


            if yer is not None:

                # ax_1.plot(x[jj], yer_datas_high, color=coli,
                #               label=y_names[jj] + ' + SE', linestyle='--')
                # ax_1.plot(x[jj], yer_datas_low, color=coli,
                #               label=y_names[jj] + ' - SE', linestyle='--')
                if marker:
                    markerstyli = marker_style_dict[jj]
                    if markerstyli and not style:
                        capsizi = default_plot_specs['cap_size']
                    else:
                        capsizi = None
                    ax.errorbar(x[jj],y[jj], yer[jj], color=coli,marker=markerstyli,
                                  markersize=default_plot_specs['marker_size'],capsize=capsizi,
                                label=y_names[jj],
                                  linewidth=default_plot_specs['linewidth'],linestyle=style)
                else:
                    yer_datas_high = [y_i + y_er_i for y_i, y_er_i in zip(y[jj], yer[jj])]
                    yer_datas_low = [y_i - y_er_i for y_i, y_er_i in zip(y[jj], yer[jj])]
                    ax.plot(x[jj], yer_datas_high, color=coli, linestyle='--')
                    ax.plot(x[jj], yer_datas_low, color=coli, linestyle='--')


        # Set the tick labels font
        for labeli in (ax.get_xticklabels() + ax.get_yticklabels()):
            # labeli.set_fontname('Arial')
            labeli.set_fontsize(default_plot_specs['ticksize'])



        # set axes limits
        if domain is None:
            xlow, xhigh = min(x[0]), max(x[0])
            for xx in x[1:]:
                mycopy_low = [g for g in copy.deepcopy(xx)]
                mycopy_high = [g for g in copy.deepcopy(xx)]
                mycopy_low.append(xlow)
                mycopy_high.append(xhigh)
                xlow, xhigh = min(mycopy_low), max(mycopy_high)
            extra = (xhigh-xlow)*default_plot_specs['x_scale']
            xlow -= extra
            xhigh +=extra
        elif ax_num ==0:
            xlow =xlow1
            xhigh = xhigh1
        else:
            xlow = xlow2
            xhigh = xhigh2
        #Make vertical lines
        for xfloat in vlines:
            if xlow < xfloat < xhigh:
                ax.axvline(x=xfloat,color = default_plot_specs['vlinecolor'],linestyle= default_plot_specs['vlinestyle'],linewidth=default_plot_specs['vlineswidth'])

        if not marker:
            xhigh -= 15

        if yrange is None:
            if y:
                if y[0]:
                    ylow, yhigh = min(y[0]), max(y[0])
                else:
                    ylow, yhigh = 0, 0
            else:
                ylow, yhigh = 0, 0
            for yy in y[1:]:
                ylow, yhigh = min([ylow] + yy), max([yhigh]+ yy)
            extra = (yhigh-ylow)*default_plot_specs['y_scale']
            ylow -= extra
            yhigh +=extra

        ax.set_xlim(xlow, xhigh)
        ax.set_ylim(ylow, yhigh)

        ax_num +=1


    if split:
        fig.axes[0].set_xlabel(x_label, **default_plot_specs['axis_font'])
        fig.axes[0].xaxis.set_label_coords(0.5, 0.01, transform=fig.transFigure)
        if default_plot_specs['nxticks'] > 0:
            ax_2.locator_params(axis='x', nticks=default_plot_specs['nxticks'])

        if default_plot_specs['nyticks'] > 0:
            ax_2.locator_params(axis='y', nticks=default_plot_specs['nyticks'])

        # This looks pretty good, and was fairly painless, but you can get that
        # cut-out diagonal lines look with just a bit more work. The important
        # thing to know here is that in axes coordinates, which are always
        # between 0-1, spine endpoints are at these locations (0,0), (0,1),
        # (1,0), and (1,1).  Thus, we just need to put the diagonals in the
        # appropriate corners of each of our axes, and so long as we use the
        # right transform and disable clipping.

        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax_1.transAxes, color='k', clip_on=False)
        ax_1.plot((1-d,1+d), (-d,+d), **kwargs)  # top-left diagonal
        ax_1.plot((1-d,1+d),(1-d,1+d), **kwargs)  # top-right diagonal

        kwargs.update(transform=ax_2.transAxes)  # switch to the bottom axes
        ax_2.plot((-d,+d), (1-d,1+d), **kwargs)  # bottom-left diagonal
        ax_2.plot((-d,+d), (-d,+d), **kwargs)  # bottom-right diagonal
    else:
        ax_1.set_xlabel(x_label, **default_plot_specs['axis_font'])

    ax_1.set_ylabel(y_label, **default_plot_specs['axis_font'])

    while under_text:
        texti = under_text.pop(0)
        plt.figtext(0.08, text_heights.pop(0), texti, default_plot_specs['undertext_font'])

    if text:
        ax_1.text(default_plot_specs['text_loc'][0], default_plot_specs['text_loc'][1], text,
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax_1.transAxes,
                color=default_plot_specs['text_color'], fontsize=default_plot_specs['text_size'])

    if ynames:
        handles, labels = ax_1.get_legend_handles_labels()
        lgd = ax_1.legend(handles, labels, loc=default_plot_specs['legend_anchor'],
                          bbox_to_anchor=default_plot_specs['legend_loc'],
                          prop = default_plot_specs['legend_font'],ncol=n_legend_columns)
        if legend_title:
            lgd.set_title(legend_title,prop=default_plot_specs['legend_font'])


    if default_plot_specs['nxticks'] > 0:
        print(default_plot_specs['nxticks'])
        ax_1.locator_params(axis='x', nticks=default_plot_specs['nxticks'])

    if default_plot_specs['nyticks'] > 0:
        ax_1.locator_params(axis='y', nticks=default_plot_specs['nyticks'])

    plt.grid('off')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir,'%s.png' % the_label)

    if ynames:
        plt.savefig(save_dir, bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        plt.savefig(save_dir, bbox_inches='tight')  # ,bbox_extra_artists=(lgd,))

    plt.close('all')



def plot_many_y_axes(x, y, yer=None, xlabel = None, ylabels = None, label = None, domain=None,
                     yrange = None, undertext =None, savedir = None, plotspecs = None):
    """
    Plots multiple y against x, with every y having it's own axis. Saves the plot.

    Parameters:
        x: (list) A list [x_1,x_2,...] with every x_i a list of x values
        y: (list) A list [y_1, y_2...] with every y_i a list of y values
        yer: (list) A list [yer_1, yer_2,...] with every yer_i a list of the standard error on y_i
        xlabel: (string) The name of the x-axis
        ylabels: (list) A list of strings - i-th element a label for the i-th y-axis
        label: (string) A label for the plot
        domain: (list) Form [xlow, xhigh] - the domain of the plot
        yrange: (list) Form [ylow, yhigh] - the yrange of the plot
        undertext: (list) List of strings to be placed beneath the plot
        savedir: (string) The name of the directory to save the graph in
        plotspecs: (dict) A dictionary to set the plot specifications if they differ from default -
            Possible keys:
                fsize: (tup) Gives dimensions of the plot (e.g. (11,4))
                linewidth: (float) Width of the lines in the plot
                axis_font: (dict) (e.g. {'fontname': 'Arial', 'size': '18'})
                ticksize: (float) Size of the ticks
                dpi: (float) Quality of the image
                undertext_font: (dict) Font of undertext
                    (e.g. {'color': 'black', 'weight': 'roman', 'size': 'x-small'})
                x_scale: (float) The x-axis is (1+2x_scale)*(xmax -xmin) wide, if domain is None
                y_scale: (float) The y-axis is (1+2y_scale)*(ymax -ymin) high, if domain is None
    """
    if savedir is None:
        save_dir = os.getcwd()
    else:
        save_dir = savedir

    number_y = len(y)

    default_plot_specs = copy.deepcopy(default_plot_specs_all)

    text_heights = [-0.023, -0.069, -0.115,-0.161]

    if plotspecs is not None:
        for stat in list(default_plot_specs.keys()):
            if stat in plotspecs:
                default_plot_specs[stat] = plotspecs[stat]

    the_label = ''

    if domain is not None:
        xlow = domain[0]
        xhigh = domain[1]
    if yrange is not None:
        ylow = yrange[0]
        yhigh = yrange[1]
    if xlabel is None:
        x_label = 'x'
    else:
        x_label = xlabel
    if ylabels is None:
        y_labels = ['y_' + str(j+1) for j in range(number_y)]
        the_label = 'y_1_' +str(number_y) +'_'
    else:
        y_labels = ylabels
        for yy in y_labels:
            the_label += yy[:4] + '_'
    if label is None:
        the_label = 'Many_' + the_label + 'vs_' +x_label
    else:
        the_label = label

    under_text = []
    if undertext is not None:
        under_text = undertext[:]


    plt.clf()

    fig = plt.figure(figsize=default_plot_specs['fsize'], dpi=default_plot_specs['dpi'])
    ax_1 = fig.add_subplot(111)

    axes = [ax_1]
    for _ in y[1:]:
        # Twin the x-axis twice to make independent y-axes.
        axes.append(ax_1.twinx())
    extra_ys = len(axes[2:])

    # Make some space on the right side for the extra y-axes.
    if extra_ys > 0:
        temp = 0.85
        if extra_ys == 1:
            temp = 0.88
        elif extra_ys <= 2:
            temp = 0.77  # 0.75
        elif extra_ys <= 3:
            temp = 0.6
        # elif extra_ys<=4:
        #     temp = 0.6
        if extra_ys > 3:
            print('you are being ridiculous')
        fig.subplots_adjust(right=temp)
        right_additive = (0.99 - temp) / float(extra_ys)  # (0.98-temp)/float(extra_ys)
    # Move the last y-axis spine over to the right by x% of the width of the axes
    ii = 1.
    for ax in axes[2:]:
        ax.spines['right'].set_position(('axes', 1. + right_additive * ii))
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        ax.yaxis.set_major_formatter(matplotlib.ticker.OldScalarFormatter())
        ii += 1.
    # To make the border of the right-most axis visible, we need to turn the frame
    # on. This hides the other plots, however, so we need to turn its fill off.


    colors = cm.rainbow(np.linspace(0, 1, number_y))

    for jj in range(number_y):
        coli = colors[jj]
        if jj == 0:
            axes[jj].plot(x[jj], y[jj], color=coli, linewidth=default_plot_specs['linewidth'])
        else:
            axes[jj].plot(x[jj], y[jj], color=coli, linewidth=default_plot_specs['linewidth'])
        if yer is not None:
            yer_datas_high = [y_i + y_er_i for y_i, y_er_i in zip(y[jj], yer[jj])]
            yer_datas_low = [y_i - y_er_i for y_i, y_er_i in zip(y[jj], yer[jj])]

            axes[jj].plot(x[jj], yer_datas_high, color=coli, linestyle='--')
            axes[jj].plot(x[jj], yer_datas_low, color=coli, linestyle='--')
        axes[jj].set_ylabel(y_labels[jj], color=coli, **default_plot_specs['axis_font'])
        axes[jj].spines['right'].set_color(coli)

        # Set the tick labels font
        for labeli in (axes[jj].get_xticklabels() + axes[jj].get_yticklabels()):
            # labeli.set_fontname('Arial')
            labeli.set_fontsize(default_plot_specs['ticksize'])

    ax_1.set_xlabel(x_label, **default_plot_specs['axis_font'])


    # set axes limits
    if domain is None:
        xlow, xhigh = min(x[0]), max(x[0])
        for xx in x[1:]:
            xlow, xhigh = min([xlow]+ xx), max([xhigh] + xx)
        extra = (xhigh-xlow)*default_plot_specs['x_scale']
        xlow -= extra
        xhigh +=extra

    if yrange is None:
        if y[0] is not None:
            ylow, yhigh = min(y[0]), max(y[0])
        else:
            ylow, yhigh = 0, 0
        for yy in y[1:]:
            ylow, yhigh = min([ylow] + yy), max([yhigh]+ yy)
        extra = (yhigh-ylow)*default_plot_specs['y_scale']
        ylow -= extra
        yhigh +=extra

    for axi in axes:
        axi.set_xlim(xlow, xhigh)
        axi.set_ylim(ylow, yhigh)

    while under_text:
        text = under_text.pop(0)
        plt.figtext(0.08, text_heights.pop(0), text, default_plot_specs['undertext_font'])

    # plt.grid('on')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir,'%s.png' % the_label)

    plt.savefig(save_dir, bbox_inches='tight')  # ,bbox_extra_artists=(lgd,))

    plt.close('all')


def plot_y(x, y, yer=None, xlabel = None, ylabel = None, yname = None, label = None, domain=None,
           yrange = None, undertext =None, savedir = None, marker = None, plotspecs = None, vlines=None):
    """
    Plots one y against one x. Saves the plot.

    Parameters:
        x: (list) A list of the x-values
        y: (list) A list of the y-values
        yer: (list) A list of the standard error on y
        xlabel: (string) The name of the x-axis
        ylabel: (float) The name of the y-axis
        yname: (float) Name of the thing being plotted
        label: (string) A label for the plot
        domain: (list) Form [xlow, xhigh] - the domain of the plot
        yrange: (list) Form [ylow, yhigh] - the yrange of the plot
        undertext: (list) List of strings to be placed beneath the plot
        savedir: (string) The name of the directory to save the graph in
        vlines: (list of floats or float) Values at which a vertical line is to be plotted
        plotspecs: (dict) A dictionary to set the plot specifications if they differ from default -
            Possible keys:
                fsize: (tup) Gives dimensions of the plot (e.g. (11,4))
                linewidth: (float) Width of the lines in the plot
                axis_font: (dict) (e.g. {'fontname': 'Arial', 'size': '18'})
                ticksize: (float) Size of the ticks
                dpi: (float) Quality of the image
                undertext_font: (dict) Font of undertext
                    (e.g. {'color': 'black', 'weight': 'roman', 'size': 'x-small'})
                x_scale: (float) The x-axis is (1+2x_scale)*(xmax -xmin) wide, if domain is None
                y_scale: (float) The y-axis is (1+2y_scale)*(ymax -ymin) high, if domain is None
                legend_font: (dict) Size and font of legend writing (e.g. {'size': 8})
                legend_anchor: (string) The point on the legend that's anchored (e.g. upper right)
                legend_loc: (tup) Point on plot where the legend is anchored (e.g. (0.98, -0.1))
                vlinestyle: (string) The style of vertical line
                vlinecolor: (string) The colour of vertical lines
    """

    if yname is not None:
        ynames = [yname]
    else:
        ynames = None

    plot_many_y([x], [y], yer=yer, xlabel=xlabel, ylabel=ylabel, ynames=ynames, label=label, domain=domain,
                yrange=yrange, undertext=undertext, savedir=savedir, marker=marker, plotspecs=plotspecs)

def plot_many_y_hist(binedges, y, yer=None, xlabel = None, ylabel = None, ynames = None, label = None, domain=None,
                     yrange = None, undertext =None, savedir = None, plotspecs = None):
    """
    Plots multiple y against x, on a single set of axes. Saves the plot.

    Parameters:
        binedges: (list) A list [be_1,be_2,...] with every be_i a list of bin edge values be_
        y: (list) A list [y_1, y_2...] with every y_i a list of y values
        yer: (list) A list [yer_1, yer_2,...] with every yer_i a list of the standard error on y_i
        xlabel: (string) The name of the x-axis
        ylabel: (float) The name of the y-axis
        ynames: (list) A list of the names of the things being plotted
        label: (string) A label for the plot
        domain: (list) Form [xlow, xhigh] - the domain of the plot
        yrange: (list) Form [ylow, yhigh] - the range of the plot
        undertext: (list) List of strings to be placed beneath the plot
        savedir: (string) The name of the directory to save the graph in
        plotspecs: (dict) A dictionary to set the plot specifications if they differ from default -
            Possible keys:
                fsize: (tup) Dimensions of the plot (e.g. (11,4))
                linewidth: (float) Width of the lines in the plot
                axis_font: (dict) (e.g. {'fontname': 'Arial', 'size': '18'})
                ticksize: (float) Size of the ticks
                dpi: (float) Quality of the image (e.g. 180)
                undertext_font: (dict) Font of undertext
                    (e.g. {'color': 'black', 'weight': 'roman', 'size': 'x-small'})
                x_scale: (float) The x-axis is (1+2x_scale)*(xmax -xmin) wide, if domain is None
                y_scale: (float) The y-axis is (1+2y_scale)*(ymax -ymin) high, if domain is None
                xlog: (bool) if true plots x-axis on log scale
                legend_font: (dict) Size and font of legend writing (e.g. {'size': 8})
                legend_anchor: (string) The point on the legend that's anchored (e.g. upper right)
                legend_loc: (tup) Point on plot where the legend is anchored (e.g. (0.98, -0.1))
                alpha: (float) Opaqueness of the histogram (e.g. 0.3)
                hist_type: (string) Histogram type (e.g. bar or step)
    """
    if savedir is None:
        save_dir = os.getcwd()
    else:
        save_dir = savedir

    number_y = len(y)


    default_plot_specs = copy.deepcopy(default_plot_specs_all)
    default_plot_specs['legend_font'] = {'size': 8}
    default_plot_specs['legend_anchor'] = 'upper right'
    default_plot_specs['legend_loc'] = (0.98, -0.1)
    default_plot_specs['alpha'] = 0.3
    default_plot_specs['hist_type'] = 'bar' #'step'

    text_heights = [-0.023, -0.069, -0.115,-0.161]

    if plotspecs is not None:
        for stat in list(default_plot_specs.keys()):
            if stat in plotspecs:
                default_plot_specs[stat] = plotspecs[stat]

    the_label = ''

    if domain is not None:
        xlow = domain[0]
        xhigh = domain[1]
    if yrange is not None:
        ylow = yrange[0]
        yhigh = yrange[1]
    if xlabel is None:
        x_label = 'x'
    else:
        x_label = xlabel
    if ylabel is None:
        y_label = 'y'
        the_label = 'y_' +str(number_y) +'_'
    else:
        y_label = ylabel
        the_label += y_label[:4] +'_'
    if ynames is None:
        y_names = ['' for _ in range(number_y)]
    else:
        y_names = ynames
    if label is None:
        the_label = the_label + 'vs_' +x_label
    else:
        the_label = label

    under_text = []
    if undertext is not None:
        under_text = undertext[:]

    plt.clf()

    fig = plt.figure(figsize=default_plot_specs['fsize'], dpi=default_plot_specs['dpi'])
    ax_1 = fig.add_subplot(111)

    if default_plot_specs['xlog']:
        ax_1.set_xscale('log')


    colors = cm.rainbow(np.linspace(0, 1, number_y))
    hist_type = default_plot_specs['hist_type']
    for jj in range(number_y):
        coli = colors[jj]
        x = [(e1 + e2) / 2.0 for e1, e2 in zip(binedges[jj][1:], binedges[jj][:-1])]
        weights = y[jj]

        ax_1.hist(x, bins=binedges[jj], weights=weights, label=y_names[jj], histtype=hist_type,
                  alpha=default_plot_specs['alpha'],color=coli)

        if yer is not None:
            #plt.errorbar(x, weights, yerr=yer[jj], fmt='none',color=coli)
            plt.errorbar(x, weights, yerr=yer[jj], fmt='none', color=coli)

    ax_1.set_ylabel(y_label, **default_plot_specs['axis_font'])


    # Set the tick labels font
    for labeli in (ax_1.get_xticklabels() + ax_1.get_yticklabels()):
        # labeli.set_fontname('Arial')
        labeli.set_fontsize(default_plot_specs['ticksize'])

    ax_1.set_xlabel(x_label, **default_plot_specs['axis_font'])

    # set axes limits
    if domain is None:
        xlow, xhigh = min(binedges[0]), max(binedges[0])
        for xx in binedges[1:]:
            xlow, xhigh = min([xlow]+ xx), max([xhigh] + xx)
        extra = (xhigh-xlow)*default_plot_specs['x_scale']
        xlow -= extra
        xhigh +=extra

    if yrange is None:
        if yer is not None:
            ycombhigh = [y[0][i]+ yer[0][i] for i in range(len(y[0]))]
            ycomblow = [y[0][i] - yer[0][i] for i in range(len(y[0]))]
        else:
            ycombhigh = y[0]
            ycomblow= y[0]
        ylow, yhigh = min(ycomblow), max(ycombhigh)
        if yer is None:
            for yy in y[1:]:
                ylow, yhigh = min([ylow] + yy), max([yhigh]+ yy)
        else:
            for yy, yyer in zip(y[1:],yer[1:]):
                ycombhigh = [yy[i] + yyer[i] for i in range(len(yy))]
                ycomblow = [yy[i] - yyer[i] for i in range(len(yy))]
                ylow, yhigh = min([ylow] + ycomblow), max([yhigh] + ycombhigh)
        extra = (yhigh-ylow)*default_plot_specs['y_scale']
        ylow -= extra
        yhigh +=extra

    ax_1.set_xlim(xlow, xhigh)
    ax_1.set_ylim(ylow, yhigh)

    while under_text:
        text = under_text.pop(0)
        plt.figtext(0.08, text_heights.pop(0), text, default_plot_specs['undertext_font'])

    if ynames:
        handles, labels = ax_1.get_legend_handles_labels()
        lgd = ax_1.legend(handles, labels, loc=default_plot_specs['legend_anchor'],
                          bbox_to_anchor=default_plot_specs['legend_loc'],
                          prop = default_plot_specs['legend_font'])

    #plt.grid('on')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir,'%s.png' % the_label)

    if ynames:
        plt.savefig(save_dir, bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        plt.savefig(save_dir, bbox_inches='tight')  # ,bbox_extra_artists=(lgd,))

    plt.close('all')

def plot_many_y_hist_and_many_y(binedges, y_hist, yer_hist=None, x=None, y=None, xlabel = None, ylabel = None, ynames_hist = None, ynames=None, label = None, domain=None,
                                yrange = None, undertext =None, savedir = None, plotspecs = None):
    """
    Plots multiple y against x, on a single set of axes. Saves the plot.

    Parameters:
        binedges: (list) A list [be_1,be_2,...] with every be_i a list of bin edge values be_
        y_hist: (list) A list [y_1, y_2...] with every y_i a list of y values
        yer_hist: (list) A list [yer_1, yer_2,...] with every yer_i a list of the standard error on y_i
        x: (list) A list [x_1,x_2,...] with every x_i a list of x values
        y: (list) A list [y_1, y_2...] with every y_i a list of y values
        xlabel: (string) The name of the x-axis
        ylabel: (float) The name of the y-axis
        ynames_hist: (list) A list of the names of the things being histogram plotted
        ynames: (list) A list of the names of the things being plotted
        label: (string) A label for the plot
        domain: (list) Form [xlow, xhigh] - the domain of the plot
        yrange: (list) Form [ylow, yhigh] - the range of the plot
        undertext: (list) List of strings to be placed beneath the plot
        savedir: (string) The name of the directory to save the graph in
        plotspecs: (dict) A dictionary to set the plot specifications if they differ from default -
            Possible keys:
                fsize: (tup) Dimensions of the plot (e.g. (11,4))
                linewidth: (float) Width of the lines in the plot
                axis_font: (dict) (e.g. {'fontname': 'Arial', 'size': '18'})
                ticksize: (float) Size of the ticks
                dpi: (float) Quality of the image (e.g. 180)
                undertext_font: (dict) Font of undertext
                    (e.g. {'color': 'black', 'weight': 'roman', 'size': 'x-small'})
                x_scale: (float) The x-axis is (1+2x_scale)*(xmax -xmin) wide, if domain is None
                y_scale: (float) The y-axis is (1+2y_scale)*(ymax -ymin) high, if domain is None
                xlog: (bool) if true plots x-axis on log scale
                legend_font: (dict) Size and font of legend writing (e.g. {'size': 8})
                legend_anchor: (string) The point on the legend that's anchored (e.g. upper right)
                legend_loc: (tup) Point on plot where the legend is anchored (e.g. (0.98, -0.1))
                alpha: (float) Opaqueness of the histogram (e.g. 0.3)
                hist_type: (string) Histogram type (e.g. bar or step)
    """
    if savedir is None:
        save_dir = os.getcwd()
    else:
        save_dir = savedir

    number_y_hist = len(y_hist)

    if y is not None:
        number_y = len(y)


    default_plot_specs = copy.deepcopy(default_plot_specs_all)
    default_plot_specs['legend_font'] = {'size': 8}
    default_plot_specs['legend_anchor'] = 'upper right'
    default_plot_specs['legend_loc'] = (0.98, -0.1)
    default_plot_specs['alpha'] = 0.3
    default_plot_specs['hist_type'] = 'bar' #'step'

    # if not ynames:
    #     height = 0.046
    #     highest = -0.023
    #     xundertext_pos = 0.08
    # else:
    #     height = 0.2
    #     highest = -0.023
    #     xundertext_pos = 0.6
    height = 0.046
    highest = -0.023
    xundertext_pos = 0.08
    text_heights = [highest - i * height for i in range(6)]

    if plotspecs is not None:
        for stat in list(default_plot_specs.keys()):
            if stat in plotspecs:
                default_plot_specs[stat] = plotspecs[stat]

    the_label = ''

    if domain is not None:
        xlow = domain[0]
        xhigh = domain[1]
    if yrange is not None:
        ylow = yrange[0]
        yhigh = yrange[1]
    if xlabel is None:
        x_label = 'x'
    else:
        x_label = xlabel
    if ylabel is None:
        y_label = 'y'
        the_label = 'y_' +str(number_y_hist) +'_'
    else:
        y_label = ylabel
        the_label += y_label[:4] +'_'
    if ynames_hist is None:
        y_names_hist = ['' for _ in range(number_y_hist)]
    else:
        y_names_hist = ynames_hist
    if label is None:
        the_label = the_label + 'vs_' +x_label
    else:
        the_label = label

    under_text = []
    if undertext is not None:
        under_text = undertext[:]

    plt.clf()

    fig = plt.figure(figsize=default_plot_specs['fsize'], dpi=default_plot_specs['dpi'])
    ax_1 = fig.add_subplot(111)

    if default_plot_specs['xlog']:
        ax_1.set_xscale('log')


    colors = cm.rainbow(np.linspace(0, 1, number_y_hist))
    hist_type = default_plot_specs['hist_type']
    for jj in range(number_y_hist):
        coli = colors[jj]
        x_hist = [(e1 + e2) / 2.0 for e1, e2 in zip(binedges[jj][1:], binedges[jj][:-1])]
        weights = y_hist[jj]

        ax_1.hist(x_hist, bins=binedges[jj], weights=weights, label=y_names_hist[jj], histtype=hist_type,
                  alpha=default_plot_specs['alpha'],color=coli)

        if yer_hist is not None:
            #plt.errorbar(x, weights, yerr=yer[jj], fmt='none',color=coli)
            plt.errorbar(x_hist, weights, yerr=yer_hist[jj], fmt='none', color=coli)


    if y is not None:
        thickness = default_plot_specs['linewidth']
        mycolors = cm.seismic(np.linspace(0, 0.9, number_y))
        for jj in range(number_y):
            if ynames is not None:
                ax_1.plot(x[jj], y[jj],label=ynames[jj],linewidth=thickness,color=mycolors[jj])
            else:
                ax_1.plot(x[jj], y[jj],linewidth=thickness)

    ax_1.set_ylabel(y_label, **default_plot_specs['axis_font'])


    # Set the tick labels font
    for labeli in (ax_1.get_xticklabels() + ax_1.get_yticklabels()):
        # labeli.set_fontname('Arial')
        labeli.set_fontsize(default_plot_specs['ticksize'])

    ax_1.set_xlabel(x_label, **default_plot_specs['axis_font'])

    # set axes limits
    if domain is None:
        xlow, xhigh = min(binedges[0]), max(binedges[0])
        for xx in binedges[1:]:
            xlow, xhigh = min([xlow]+ xx), max([xhigh] + xx)
        extra = (xhigh-xlow)*default_plot_specs['x_scale']
        xlow -= extra
        xhigh +=extra

    #now
    index_list = []
    for binedge_set in binedges:
        indices = []
        i = 0
        for xx in binedge_set:
            if xlow < xx < xhigh:
                indices.append(i)
            i += 1
        index_list.append([min(indices),max(indices)])

    if yrange is None:
        if yer_hist is not None:
            ycombhigh = [y_hist[0][i]+ yer_hist[0][i] for i in range(len(y_hist[0]))]
            ycomblow = [y_hist[0][i] - yer_hist[0][i] for i in range(len(y_hist[0]))]
        else:
            ycombhigh = y_hist[0]
            ycomblow= y_hist[0]
        ylow, yhigh = min(ycomblow), max(ycombhigh)
        if yer_hist is None:
            for yy in y_hist[1:]:
                ylow, yhigh = min([ylow] + yy), max([yhigh]+ yy)
        else:
            for yy, yyer in zip(y_hist[1:],yer_hist[1:]):
                ycombhigh = [yy[i] + yyer[i] for i in range(len(yy))]
                ycomblow = [yy[i] - yyer[i] for i in range(len(yy))]
                ylow, yhigh = min([ylow] + ycomblow), max([yhigh] + ycombhigh)
        extra = (yhigh-ylow)*default_plot_specs['y_scale']
        ylow -= extra
        yhigh +=extra

    ax_1.set_xlim(xlow, xhigh)
    ax_1.set_ylim(ylow, yhigh)


    # ynames = False
    # ynames_hist = False
    if ynames_hist or ynames:
        handles, labels = ax_1.get_legend_handles_labels()
        lgd = ax_1.legend(handles, labels, loc=default_plot_specs['legend_anchor'],
                          bbox_to_anchor=default_plot_specs['legend_loc'],
                          prop = default_plot_specs['legend_font'])

    while under_text:
        text = under_text.pop(0)
        figtext = plt.figtext(xundertext_pos, text_heights.pop(0), text, default_plot_specs['undertext_font'])


    #plt.grid('on')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir,'%s.png' % the_label)

    if ynames_hist or ynames:
        if undertext:
            plt.savefig(save_dir, bbox_extra_artists=(lgd,figtext,), bbox_inches='tight')
        else:
            plt.savefig(save_dir, bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        plt.savefig(save_dir, bbox_inches='tight')  # ,bbox_extra_artists=(lgd,))

    plt.close('all')

