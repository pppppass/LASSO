import numpy
import matplotlib.pyplot as plt

def errfun(x1, x2):
    return numpy.linalg.norm(x1 - x2) / (1. + numpy.linalg.norm(x1))

def l1_gt_wrapper_gen(u):
    def l1_gt_wrapper(x0, A, b, mu, **opts):
        return u, {"solution": u}
    return l1_gt_wrapper

notebook_config = {
    "name": ["Name", "{}"],
    "time": ["Time", "{:.5f}"],
    "setup_time": ["Setup time", "{:.5f}"],
    "solve_time": ["Solve time", "{:.5f}"],
    "vars": ["Variables", "{}"],
    "iters": ["Iterations", "{}"],
    "loss": ["Loss", "{:.5e}"],
    "check_loss": ["Check loss", "{:.5e}"],
    "approximation_loss": ["Approx loss", "{:.5e}"],
    "regularization": ["Regularization", "{:.5e}"],
    "error_xx": ["Error to known", "{:.5e}"],
    "error_gt": ["Error to GT", "{:.5e}"],
}

LaTeX_config = [{
    "name": ["", "{}"],
    "time": ["time (\Si{\second})", "{:.3f}"],
    "setup_time": ["setup time (\Si{\second})", "{:.3f}"],
    "solve_time": ["solve time (\Si{\second})", "{:.3f}"],
    "vars": ["variables", "{}"],
    "iters": ["iterations", "{}"],
},{
    "name": ["", "{}"],
    "check_loss": ["primal objective", "{:.5e}"],
    "approximation_loss": ["approximation loss", "{:.5e}"],
    "error_xx": ["error to known", "{:.3e}"],
    "error_gt": ["error to GT", "{:.3e}"],
}]

def format_notebook(out, config):
    for key, val in config.items():
        if key in out:
            name = val[0]
            rep = val[1].format(out[key])
            print("{0}: {1}".format(name, rep))

def format_LaTeX_piece(out, config, heading=True):
    if heading:
        format_ind = "|".join(["c"]*len(config))
        print(r"\begin{{tabular}}{{|{}|}}".format(format_ind))
        print(r"\hline")
        string = " & ".join([
            val[0]
            for key, val in config.items()
        ])
        string += r" \\ \hline"
        print(string)
    
    string = " & ".join([
        val[1].format(out[key])
        if key in out
        else "NA"
        for key, val in config.items()
    ])
    string += r" \\ \hline"
    print(string)

def format_LaTeX_end():
    print(r"\end{tabular}")

def format_LaTeX(out_list, config):
    first = True
    for out in out_list:
        format_LaTeX_piece(out, config, heading=first)
        first = False
    format_LaTeX_end()

def draw_loss_curve(out, label, log=False):
    
    for key, val in label.items():
        array = out[key]
        
        if log:
            array = numpy.log10(array)
        
        plt.plot(array, label=val)
    
    plt.legend()

    plt.show()
    
    return None

class Stat(object):
    def __init__(self):
        self.stat = []
    
    def __call__(self, out):
        self.append(out)
    
    def append(self, out):
        self.stat.append(out)
    
    def pop(self):
        self.stat.pop()
    
    def notebook_last(self, config=notebook_config):
        format_notebook(self.stat[-1], config)
    
    def loss_curve_last(self, label={"real_loss": "Real loss", "formal_loss": "Modified loss"}, log=False):
        draw_loss_curve(self.stat[-1], label=label, log=log)
    
    def LaTeX_all(self, config_list=LaTeX_config):
        format_LaTeX(self.stat, config_list[0])
        format_LaTeX(self.stat, config_list[1])
