import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# These lines are required to make fonts the right form for adobe illustrator
# import matplotlib
# matplotlib.rcParams["pdf.fonttype"] = 42
# matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

fontsize_text = 22
fontsize_title = 22
fontsize_ticks = 20
fontsize_cbar = 16
fontsize_legend = 18
saveextension = ".png"
transperantFlag = False


def addAxis(thisfig, n1, n2, maxnumaxis=""):
    axlist = []
    if maxnumaxis:
        counterval = maxnumaxis
    else:
        counterval = n1 * n2

    for i in range(counterval):
        axlist.append(thisfig.add_subplot(n1, n2, i + 1))

    return axlist


def addColorbar(thisfig, thisax, thisim, cbartitle="", fontsize_cbar=fontsize_cbar, fontsize_ticks=fontsize_ticks):
    divider = make_axes_locatable(thisax)
    cax = divider.append_axes("right", size="8%", pad=0.05)
    cbar = thisfig.colorbar(thisim, cax=cax, orientation="vertical")

    # option to change colorbar to horizontal bottom or other location can be done as so
    # cax = divider.append_axes("bottom", size="8%", pad=0.05)
    # cbar = thisfig.colorbar(thisim, cax=cax, orientation="horizontal")

    cbar.formatter.set_powerlimits((0, 0))
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(cbartitle, rotation=90, fontsize=fontsize_cbar)

    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(fontsize_ticks)

    return


def formatPlots(
    thisfig,
    thisax,
    imhandle,
    xlabel="",
    ylabel="",
    title="",
    rmvxLabel=False,
    rmvyLabel=False,
    addcolorbar=False,
    cbartitle="",
    setxlim=[],
    setylim=[],
    addlegend=False,
    fontsize_text=fontsize_text,
    fontsize_title=fontsize_title,
    fontsize_ticks=fontsize_ticks,
    fontsize_cbar=fontsize_cbar,
    fontsize_legend=fontsize_legend,
    setAspect="auto",
):  # Pass figure and axis to set common formatting options

    thisax.set_xlabel(xlabel, fontsize=fontsize_text)
    thisax.set_ylabel(ylabel, fontsize=fontsize_text)
    thisax.set_title(title, fontsize=fontsize_title)

    if addcolorbar:
        addColorbar(thisfig, thisax, imhandle, cbartitle, fontsize_cbar=fontsize_cbar, fontsize_ticks=fontsize_ticks)
    else:
        # This is useful to change the axis size such that the axis with and without colorbar is the same shape
        divider2 = make_axes_locatable(thisax)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        cax2.axis("off")

    if rmvxLabel:
        thisax.set_xticklabels([""])
        thisax.set_xlabel("")

    if rmvyLabel:
        thisax.set_yticklabels([""])
        thisax.set_ylabel("")

    # update fontsize for labels and ticks
    for item in thisax.get_xticklabels() + thisax.get_yticklabels():
        item.set_fontsize(fontsize_ticks)
        plt.rcParams.update({"font.size": fontsize_ticks})

    if setxlim:
        thisax.set_xlim(setxlim[0], setxlim[1])
        # thisax.set_xbound(lower=setxlim[0], upper=setxlim[1])
    if setylim:
        thisax.set_ylim(setylim[0], setylim[1])
        # thisax.set_ybound(lower=setylim[0], upper=setylim[1])

    if addlegend:
        legend = thisax.legend(fontsize=fontsize_legend)

    # Set aspect ratio
    thisax.set_aspect(setAspect)

    return
