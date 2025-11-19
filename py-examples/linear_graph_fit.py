from batfloman_praktikum_lib import Measurement, graph_fit, DataCluster, rel_path, graph

file = rel_path("../daten/example1.csv", __file__)
data = DataCluster.load_csv(file, section="linear-fit")

# remove faulty data
taken_data = DataCluster(data[:-1])
not_taken = DataCluster(data[-1:])

for row in taken_data:
    row["x"] = Measurement(row["x"], .2)

print(taken_data)

# fit:
# currently only least-squares-fit
# => does not take x-errors into account!
fit_res = graph_fit.Linear.on_data(taken_data, "x", "y");

# you could try `orthogonal distance regression`
# => takes x-errors into account

# from scipy.odr import ODR, Model, RealData

fit_res_odr = graph_fit.Linear.odr_fit(
    x = taken_data.values("x"),
    y = taken_data.values("y"),
    xerr = taken_data.errors("x"),
    yerr = taken_data.errors("y")
)
# idk, maybe gives better/worse values...

# ==================================================
# plotting

plot = graph.create_plot(figsize=(8,5))
# alternative, the mathplotlib.pyplot (plt) is exposed:
# plot = graph.plt.subplots()

fig, ax = plot
ax.grid(0.5)

graph.scatter_data(taken_data, "x", "y", plot=plot, with_error=True, label="taken values")
graph.scatter_data(not_taken, "x", "y", plot=plot, with_error=True, color="red", label="not taken")

# change_viewport resets the x,y-limits
line_res = graph.plot_func(fit_res, plot=plot, with_error=True, change_viewport=False, label="fit");
line_res_odr = graph.plot_func(fit_res_odr, plot=plot, with_error=True, change_viewport=False, label="odr fit");

# access to the ax.plot line result:
# line_res.line.remove()

# or the fill_area (if with_error is True)
# if line_res.fill:
#     line_res.fill.remove()

ax.legend(loc="upper left", bbox_to_anchor=(1,1))
graph.plt.tight_layout()

graph.plt.show()
