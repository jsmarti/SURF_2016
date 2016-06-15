"""
Re-writing the Pareto class from scratch.

"""


import numpy as np
import scipy.stats
from scipy.stats import norm
import design
import fem_wire_problem
from . import mrange
import GPy
import copy
import os
import tempfile
import subprocess
import shutil
import seaborn as sns

def get_idx_of_observed_pareto_front_2d(Y):
    """
    Fast algorithm for the pareto front for the case of two objectives.
    """
    idx = np.lexsort(Y.T)
    idxp = [idx[0]]
    yp = Y[idx[0], 0]
    for i in idx[1:]:
        if Y[i, 0] >= yp:
            continue
        yp = Y[i, 0]
        idxp.append(i)
    idxp = np.array([i for i in idxp])
    return idxp


def get_idx_of_observed_pareto_front(Y):
    """
    Look at ``Y`` and return the indexes of the rows that are not
    dominated.
    """
    if Y.shape[1] == 2:
        return get_idx_of_observed_pareto_front_2d(Y)
    idx_to_be_eliminated = set()
    num_obj = Y.shape[1]
    num_obs = Y.shape[0]
    for i in xrange(num_obs):
        for j in xrange(i + 1, num_obs):
            vij = Y[i, :] - Y[j, :]
            if np.all(vij > 0.):
                idx_to_be_eliminated.add(i)
            elif np.all(vij < 0.):
                idx_to_be_eliminated.add(j)
    all_idx = set(np.arange(num_obs))
    idx_to_be_kept = all_idx.difference(idx_to_be_eliminated)
    return np.array([i for i in idx_to_be_kept])


def is_dominated(y, Y):
    """
    Test if ``y`` is dominated by any point in ``Y``.
    """
    for j in xrange(Y.shape[0]):
        if np.all(Y[j, :]>y):
            return False
    return True

def is_dominated_lplus(y,Y):
    """
    Test if ``y`` is dominated by any point in ``Y``
    based on the criteira required to select a cell
    that contributes to ``L+``.
    """
    for j in xrange(Y.shape[0]):
        if np.all(Y[j, :]<=y):
            return True
    return False


def strictly_dominates_any(y, Y):
    """
    Test if ``y`` strictly dominates some point of P.
    """
    for j in xrange(Y.shape[0]):
        if np.all(y < Y[j, :]):
            return True
    return False


def plot_pareto(Y, ax=None, style='-',
                color='r', linewidth=2,
                max_obj=None):
    """
    Plot the pareto front.
    """
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
    if max_obj is None:
        m = np.max(Y, axis=0)
        max_obj = m + .5 * m
    m = Y.shape[1]
    assert m == 2, 'Only works with 2 objectives.'
    idx = get_idx_of_observed_pareto_front_2d(Y)
    Y = Y[idx, :]
    n = Y.shape[0]
    ax.plot([max_obj[0], Y[0, 0]],
            [Y[0, 1], Y[0, 1]], style, color=color, linewidth=linewidth)
    for i in xrange(n-1):
        ax.plot([Y[i, 0], Y[i, 0], Y[i + 1, 0]],
                [Y[i, 1], Y[i + 1, 1], Y[i + 1, 1]], style,
                color=color,
                linewidth=linewidth)
    ax.plot([Y[-1, 0], Y[-1, 0]],
            [Y[-1, 1], max_obj[1]], style, color=color, linewidth=linewidth)
    return ax.get_figure(), ax


def compute_sorted_list_of_pareto_points(Y, y_ref):
    """
    Compute and return the sorted list of all the i-th coordingates of a
    set of Pareto points.

    This is the ``b`` of Emerich (2008). See page 5.
    """
    m = Y.shape[1]
    return np.concatenate([[[-np.inf for _ in xrange(m)]],
                           np.sort(Y, axis=0),
                           y_ref[None, :],
                           [[np.inf for _ in xrange(m)]]], axis=0)

def write_input_file(in_file,x,samples,out_file_name):
    """
    writes the input parallel code using mpi4py.
    """
    with open(in_file,'w') as fd:
        fd.write("""   
import os
import sys
import copy
import numpy as np
import math

#from _function_evaluation_wrapper import FunctionEvaluationWrapper
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.append("/home/ppandit/code/pydes/code/")
import fem_wire_problem
from mpi4py import MPI as mpi
class ObjFunc(object):
    def f1(self,x):
        y = 0
        for _ in xrange(self.n_samp):
            x1 = copy.copy(x)
            xi_1 = self.sigma1*np.random.randn((np.size(x1)/2),)
            xi_2 = self.sigma2*np.random.randn((np.size(x1)/2),)
            xi = np.concatenate((xi_1,xi_2))
            x1 = x1 + xi
            b1 = 15. * x1[0] - 5.
            b2 = 15. * x1[1]
            k = (b2 - 5.1 / 4. / math.pi ** 2 * b1 ** 2 + 5. / math.pi * b1 - 6.) ** 2. \
            + 10. * ((1. - 1. / 8. / math.pi) * math.cos(b1) + 1.)
            y = y + k    
        return float(y)/self.n_samp

    def f2(self,x):
        y = 0
        for _ in xrange(self.n_samp):
            x2 = copy.copy(x)
            xi_1 = self.sigma1*np.random.randn((np.size(x2)/2),)
            xi_2 = self.sigma2*np.random.randn((np.size(x2)/2),)
            xi = np.concatenate((xi_1,xi_2))
            x2 = x2 + xi
            b1 = 15. * x2[0] - 5.
            b2 = 15. * x2[1]
            k = - np.sqrt(np.abs((10.5 - b1) * ((b1 + 5.5)) * (b2 + 0.5))) \
            - 1. / 30. * (b2 - 5.1 / 4. / math.pi ** 2 * b1 ** 2 - 6.) ** 2 \
            - 1. / 3. * ((1. - 1. / 8. / math.pi) * math.cos(b1) + 1.)
            y = y + k
        return float(y)/self.n_samp

    def __init__(self,sigma1=0.,sigma2=0.,n_samp=1.): 
        self.n_samp = n_samp
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def __call__(self,x):
        return self.f1(x), self.f2(x)

rank = mpi.COMM_WORLD.Get_rank()
size = mpi.COMM_WORLD.Get_size()
# Print info
if rank == 0:       # This guy is known as the root
    print '=' * 80
    print 'Collecting data for the wire problem in parallel'.center(80)
    print '=' * 80
    print 'Number of processors available:', size
# Wait for the root to print this message
mpi.COMM_WORLD.barrier()   
my_num_samples = {0:d}/size
mpi.COMM_WORLD.barrier()
if rank == 0:
    X = [{1:1.5f},{2:1.5f},{3:1.5f},{4:1.5f},{5:1.5f},{6:1.5f},{7:1.5f},{8:1.5f},{9:1.5f},{10:1.5f},{11:1.5f},{12:1.5f},{13:1.5f},{14:1.5f},{15:1.5f}]
else:
    X = None
X = mpi.COMM_WORLD.bcast(X)
my_X = X
#wrapper = ObjFunc(0,0,1)
wrapper = fem_wire_problem.FunctionEvaluationWrapper(0,0.1,1)
# This loop is actually completely independent
my_Y = []
for j in xrange(my_num_samples):
    if rank == 0:
        print 'sample ' + str((j + 1) * size).zfill(6)
    my_Y.append(wrapper(my_X))
my_Y = np.array(my_Y)
# All these outputs need to be sent to the root
all_Y = mpi.COMM_WORLD.gather(my_Y)
if rank == 0:
    y = np.vstack(all_Y)
    y = np.mean(y,axis=0)
    np.save('{16:s}', y)
""".format(samples,x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13],x[14],out_file_name))

def read_output(out_file):
    """
    reads the output stored in a particular array in the temporary directory.
    """
    #output_file = out_file +'.npy'
    y = np.load(out_file)
    return y

    
def compute_samp_avg(x,samples=20):
    """
    This method writes the file , runs the parallel code, reads the output.
    """
    tmp_dir = tempfile.mkdtemp()
    file_prefix  = os.path.join(tmp_dir,'collect_data_final')
    in_file = file_prefix + '.py'
    out_file_name = file_prefix + '.npy'
    write_input_file(in_file,x=x,samples=samples,out_file_name=out_file_name)
    cmd = ['mpiexec', '-n', '20', 'python', str(in_file)]
    #DEVNULL = open(os.devnull, 'wb')
    p = subprocess.Popen(cmd, cwd=tmp_dir)
    p.wait()
    assert p.returncode == 0
    #DEVNULL.close()

    #out_file = file_prefix + '.npy'
    out = read_output(out_file_name)
    shutil.rmtree(tmp_dir)
    return out

def get_parallel_data(x,samples,obj_true):
    """
    Computes the data in parallel for the final stage.
    """
    import os
    import sys
    #from _function_evaluation_wrapper import FunctionEvaluationWrapper
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import fem_wire_problem
    from mpi4py import MPI as mpi
    n = x.shape[0]
    Y = []
    for i in xrange(n):
        y = compute_samp_avg(x[i,:],samples)
        Y.append(y)
    return Y

class ParetoFront(object):

    """
    Description.

    :param X:   Input points - num_points x num_dim
    :param Y:   Objectives - num_points x num_obj
    """

    # Observed objectives
    Y = None

    # Input points corresponding to the observed objectives
    X = None

    # The indexes of the points that correspond to the Pareto front
    idx = None

    # The reference point used for the computation of the expected improvement
    y_ref = None

    # Sorted list of all the i-th coordinates of the Pareto front vectors
    b = None

    @property
    def Y_pareto(self):
        """
        :getter: The objectives on the Pareto front.
        """
        return self.Y_p[self.idx, :]

    @property 
    def X_pareto(self):
        """
        :getter: The design points on the Pareto front.
        """
        return self.X[self.idx, :]

    @property
    def num_obj(self):
        """
        :getter: The number of objectives.
        """
        return self.Y.shape[1]

    @property
    def num_dim(self):
        """
        :getter: The number of input dimensions.
        """
        return self.X.shape[1]

    @property
    def num_pareto(self):
        """
        :getter: The number of points on the Pareto front.
        """
        return self.idx.shape[0]

    def l(self, i):
        """
        The lower coordinates of cell i.
        """
        return self.b[i, self._all_obj_idx]

    def u(self, i):
        """
        The upper coordinates of cell i.
        """
        return self.b[i + 1, self._all_obj_idx]

    def get_cell(self, i):
        """
        Getter the upper and lower corner of a cell described by indexes i.

        The index must be a numpy array of integers.
        """
        return self.l(i), self.u(i)

    @property
    def active_cells(self):
        """
        A generator of active cells.

        A cell is active if the points that it contains dominate atleast 
        one point in the pareto frontier.
        """
        k = self.num_pareto
        for i in mrange([k+1] * self.num_obj):
            l = self.l(i)
            if not is_dominated(l, self.Y_pareto):
                yield l, self.u(i), i

    @property
    def active_cells_lplus(self):
        """
        A generator for cells that are accounted for while calculating the ``L+``
        given in Emmerich
        """
        k = self.num_pareto
        for i in mrange([k+1] * self.num_obj):
            l = self.l(i)
            if not is_dominated_lplus(l, self.Y_pareto):
                yield l, self.u(i), i

    @property
    def hypervolume_cells(self):
        """
        A generator of the cells that lie in the dominated hypervolume of the 
        pareto frontier.
        """
        k = self.num_pareto
        for i in mrange([k+1]*self.num_obj):
            l = self.l(i)
            if  is_dominated_lplus(l,self.Y_pareto):
                yield l, self.u(i), i

    @property            
    def s_minus_cells(self):
        """
        A generator of the cells that form the s_minus_cells.
        """
        k = self.num_pareto
        for i in mrange([k+1]*self.num_obj):
            l = self.l(i)
            if is_dominated_lplus(l,self.Y_pareto):
                if  np.all(self.u(i)!=self.y_ref):
                    yield l, self.u(i), i

    def get_projected_observations(self):
        """
        Projecting the observations on the surrogate model.
        """
        Y_p = (self._project(self.X))[0]
        return Y_p

    def active_cells_dominated_by_lplus(self, q):
        """
        A generator of active cells that dominated by ``q``.
        """
        for l, u, i in self.active_cells_lplus:
            if np.all(q <= l):
                yield l, u, i

    def active_cells_dominated_by(self, q):
        """
        A generator of active cells that dominated by ``q``.
        """
        for l, u, i in self.active_cells:
            if np.all(q <= l):
                yield l, u, i

    def plot_cell(self, l, u, ax, color='orange', alpha=0.25):
        """
        Plot a cell given the lower and upper bounds.

        :param l:   Lower bound.
        :param u:   Upper bound.
        :param ax:  Axis of figure.
        """
        l = l.copy()
        for i in xrange(2):
            if np.isinf(l[i]):
                mx = np.max(self.Y_p[:, i])
                mn = np.min(self.Y_p[:, i])
                l[i] = mn - 0.1 * (mx - mn)
        ax.plot([l[0], l[0]], [l[1], u[1]], 'k', linewidth=0.5)
        ax.plot([l[0], u[0]], [l[1], l[1]], 'k', linewidth=0.5)
        ax.plot([l[0], u[0]], [u[1], u[1]], 'k', linewidth=0.5)
        ax.plot([u[0], u[0]], [l[1], u[1]], 'k', linewidth=0.5)
        ax.fill_between([l[0], u[0]], [l[1], l[1]], [u[1], u[1]],
                        color=color, alpha=alpha)

    def plot_active_cells(self, ax=None):
        """
        Plot all the active cells.

        :param ax:  The axis object.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        ax.plot(self.y_ref[0], self.y_ref[1], 's')
        for l, u, _ in self.active_cells:
            self.plot_cell(l, u, ax, alpha=0.5)
        return ax.get_figure(), ax

    def plot_hypervolume_cells(self,ax=None):
        """
        Plot the cells that lie in the dominated hypervolume of the pareto frontier.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        for l, u, _ in self.hypervolume_cells:
            self.plot_cell(l, u, ax, color=sns.color_palette()[2], alpha = 0.5)
        return ax.get_figure(), ax


    def plot_pareto(self, ax=None):
        """
        Plot the Pareto front of the object.
        """
        return plot_pareto(self.Y_pareto, ax=ax, max_obj=self.y_ref)

    def __init__(self, X, Y,
                 X_design=100,
                 y_ref=None,
                 constraints_func=lambda(x): 1,
                 kernel_type=GPy.kern.RBF,
                 gp_regression_type=GPy.models.GPRegression,
                 gp_opt_num_restarts=10,
                 gp_opt_verbose=False,
                 gp_fixed_noise=1e-4,
                 debug=False,
                 max_it=50,
                 verbose=False,
                 rtol=1e-3,
                 add_at_least=10,
                 make_plots=False,
                 get_fig=None,
                 fig_prefix='moo',
                 Y_true_pareto=None,
                 denoised=True,
                 samp=20):
        assert X.ndim == 2
        self.X = X
        assert Y.ndim == 2
        assert Y.shape[0] == X.shape[0]
        self.Y = Y
        if not isinstance(X_design, int):
            assert X_design.ndim == 2
            assert X_design.shape[1] == X.shape[1]
        self.X_design = X_design
        #self.obj_funcs = obj_funcs
        #self.obj_funcs_true = obj_funcs_true
        self.constraints_func = constraints_func
        if y_ref is None:
            mx = np.max(Y, axis=0)
            mn = np.min(Y, axis=0)
            y_ref = mx + 0.1 * (mx - mn)
        self.y_ref = y_ref
        self.kernel_type = kernel_type
        self.gp_regression_type = gp_regression_type
        self.gp_opt_num_restarts = gp_opt_num_restarts
        self.gp_opt_verbose = gp_opt_verbose
        self.gp_fixed_noise = gp_fixed_noise
        self.denoised = denoised
        self.samp = samp
        self._surrogates = None
        self.verbose = verbose
        self.Y_p = self.get_projected_observations()
        self.idx = get_idx_of_observed_pareto_front(self.Y_p)
        self.b = compute_sorted_list_of_pareto_points(self.Y_pareto, y_ref)
        self._all_obj_idx = np.arange(self.num_obj)
        self._debug = debug
        self.max_it = max_it
        self.rtol = rtol
        self.add_at_least = add_at_least
        self.make_plots = make_plots
        self.get_fig = get_fig
        self.fig_prefix = fig_prefix
        self.Y_true_pareto = Y_true_pareto
        

    @property
    def surrogates(self):
        """
        Get the surrogates. Train them if this hasn't happened yet.
        """
        if self._surrogates is None:
            self.train_surrogates()
        return self._surrogates

    def train_surrogates(self):
        """
        Train the surrogates.
        """
        self._surrogates = []
        for i in xrange(self.num_obj):
            k = self.kernel_type(self.num_dim, ARD=True)
            gp = self.gp_regression_type(self.X, self.Y[:, i][:, None], k)
            if self.gp_fixed_noise is not None:
                fixed_noise = self.gp_fixed_noise * np.std(self.Y[:, i])
                gp.Gaussian_noise.variance.unconstrain()
                gp.Gaussian_noise.variance.fix(fixed_noise ** 2)
            gp.optimize_restarts(self.gp_opt_num_restarts,
                                verbose=self.gp_opt_verbose)
            self._surrogates.append(gp)

    def _project(self,X):
        """
        Project the observed data on the surrogate.
        """

        n = X.shape[0]
        m = self.num_obj
        Y_m = np.ndarray((n, m))
        Y_v = np.ndarray((n, m))
        for i in xrange(m):
            m, v = self.surrogates[i].predict(X)
            Y_m[:, i] = m.flatten()
            Y_v[:, i] = v.flatten()
        return Y_m, Y_v


    def predict(self, X):
        """
        Make predictions with the surrogates.

        Return a tuple containing the predictive mean and the predictive variance
        of each objective at each provided input point.
        Both the mean and the variance are num_points x num_obj arrays.
        """
        n = X.shape[0]
        m = self.num_obj
        Y_m = np.ndarray((n, m))
        Y_v = np.ndarray((n, m))
        for i in xrange(m):
            if self.denoised:
                if hasattr(self.surrogates[i],'likelihood') and hasattr(self.surrogates[i].likelihood,'variance'):
                    noise = self.surrogates[i].likelihood.variance
                else:
                    noise = 0.
            else:
                noise = 0.
            m, v = self.surrogates[i].predict(X)
            Y_m[:, i] = m.flatten()
            Y_v[:, i] = v.flatten() - noise
        return Y_m, Y_v

    def compute_expected_improvement(self, X_design):
        """
        Compute the expected improvement over the set of design points ``X_design``.
        """
        Y_m, Y_v = self.predict(X_design)
        # Take care of numerical errors
        Y_v[Y_v <= 0.] = 0.
        # Now loop over the design points
        n = X_design.shape[0]
        ei = np.ndarray((n, ))
        for i in xrange(n):
            if self.constraints_func(X_design[i, :]) == 0:
                ei[i] = 0
            else:
                mu = Y_m[i, :]
                sigma = np.sqrt(Y_v[i, :])
                ei[i] = self._compute_expected_improvement(mu, sigma)
        return ei

    def _compute_expected_improvement(self, mu, sigma):
        """
        Compute the expected improvement of a single design point if I give you
        its predictive mean and standard deviation.
        """
        ei = 0.
        for l, u, i in self.active_cells:
            ei += self._compute_delta(mu, sigma, l, u, i)
        return ei

    def _plot_delta(self, l, u, v, i):
        """
        Plot delta to a file.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        figname = 'delta_' + str(i) + '.pdf'
        fig, ax = self.plot_pareto()
        self.plot_cell(l, u, ax)
        ax.plot(v[0], v[1], 'd')
        for l, u, _ in self.active_cells_dominated_by(u):
            self.plot_cell(l, u, ax, color='grey')
        fig.savefig(figname)
        plt.close(fig)

    def _compute_delta(self, mu, sigma, l, u, i):
        """
        Compute the delta term of Eq. (7).
        """
        v = self._find_v(i)
        S_minus_vol = self._compute_volume_of_S_minus(u, v)
        delta_j = self._delta_j(mu, sigma, l, u, v)
        # This is Eq. (7) of the paper
        delta = np.prod(delta_j) - \
                S_minus_vol * np.prod(norm.cdf((u - mu) / sigma)
                                      - norm.cdf((l - mu) / sigma))
        if self._debug:
            self._plot_delta(l, u, v, i)
        return delta

    def _find_v(self, i):
        """
        Find the vector v, required in Eq. (8) by following the recipe described
        in the text: 
        
        "We can construct the j-th coordinate of v as follows by finding the
        first vector in the sequence l(i1,...,ij, ..., im), l(i1,...,ij+1,...,im)
        which does not strictly dominate some point of P.
        "
        """
        m = self.num_obj
        v = np.ndarray(m)
        for j in xrange(m):
            for k in xrange(1, self.num_pareto + 3):
                it = i.copy()
                it[j] += k
                l = self.l(it)
                if not strictly_dominates_any(l, self.Y_pareto):
                    v[j] = l[j]
                    break
        return v
    
    def _compute_volume_of_S_minus(self, u, v):
        """
        Compute the volume of the set S minus as defined in Fig. 3.
        """
        # First compute the volume of [u, v]
        uv_vol = np.prod(v - u)
        # Now find all the active cells that dominate u and compute their
        L_plus_vol = self._compute_volume_of_L_plus(u)
        return uv_vol - L_plus_vol

    def _compute_volume_of_L_plus(self, u):
        """
        Compute the volume of the set L plus as defined in Fig. 3.
        """
        L_plus_vol = 0.
        for ln, un, _ in self.active_cells_dominated_by_lplus(u):
            L_plus_vol += np.prod(un - ln)
        return L_plus_vol

    def _delta_j(self, mu, sigma, l, u, v):
        """
        This computes delta_j (as a vector) of Eq. (9) of the paper.
        """
        return self._Psi(v, u, mu, sigma) - self._Psi(v, l, mu, sigma)

    def _Psi(self, a, b, mu, sigma):
        """
        This computes Eq. (10) of the paper.
        """
        t = (b - mu) / sigma
        return sigma * norm.pdf(t) + (a - mu) * norm.cdf(t)

    def plot_status(self, it,final=False):
        """
        Plot the status of the algorithm.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        if self.get_fig is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = self.get_fig()
        if self.Y_true_pareto is not None:
            mx = np.max(self.Y_true_pareto, axis=0)
            mn = np.min(self.Y_true_pareto, axis=0)
            max_obj = mx + 0.1 * (mx - mn)
            min_obj = mn - 0.1 * (mx - mn)
            plot_pareto(self.Y_true_pareto, ax=ax, style='-',
                        color=sns.color_palette()[0],
                        max_obj=max_obj)
        else:
            mx = np.max(self.Y_p, axis=0)
            mn = np.min(self.Y_p, axis=0)
            max_obj = mx + 0.1 * (mx - mn)
            min_obj = mn - 0.1 * (mx - mn)
        Y_pa = self.sample_pareto_fronts()
        for y_p in Y_pa:
            plot_pareto(y_p, ax=ax, style='-',
                        color=sns.color_palette()[3],
                        linewidth=0.05,
                        max_obj=max_obj)
        if final:
            #self.Y_true_noiseless = get_parallel_data(self.X_pareto,self.samp,self.obj_funcs)
            #self.Y_true_noiseless = np.vstack(self.Y_true_noiseless)
            #self.Y_true_noiseless = np.array([self.obj_funcs_true(x) for x in self.X_pareto])
            #plot_pareto(self.Y_true_noiseless[:, :], ax=ax, style='--', color=sns.color_palette()[4], max_obj=max_obj)
            #ax.plot(self.Y_true_noiseless[:, 0], self.Y_true_noiseless[:, 1], 'd', markersize=10, color=sns.color_palette()[4])
            #ax.plot(self.Y_true_noiseless[-1, 0], self.Y_true_noiseless[-1, 1], 'o', markersize=10,color=sns.color_palette()[4])
            plot_pareto(self.Y_p[:, :], ax=ax, style='--',
                    color=sns.color_palette()[1],
                    max_obj=max_obj)
            ax.plot(self.Y_p[:, 0], self.Y_p[:, 1], 'd', markersize=5, color=sns.color_palette()[1])
        else:
            plot_pareto(self.Y_p[:-1, :], ax=ax, style='--',
                        color=sns.color_palette()[1], max_obj=max_obj)
            ax.plot(self.Y_p[:-1, 0], self.Y_p[:-1, 1], 'd', color=sns.color_palette()[1], markersize=10)
            ax.plot(self.Y_p[-1, 0], self.Y_p[-1, 1], 'o', markersize=10,
                    color=sns.color_palette()[2])
            #self.plot_active_cells(ax=ax)
            #self.plot_hypervolume_cells(ax=ax)
        ax.set_xlim(min_obj[0], max_obj[0])
        ax.set_ylim(min_obj[1], max_obj[1])
        ax.set_xlabel('Objective 1',fontsize=14)
        ax.set_ylabel('Objective 2',fontsize=14)
        figname = self.fig_prefix + '_' + str(it).zfill(len(str(self.max_it))) \
                  + '.pdf'
        if self.verbose:
            print '\t> writing:', figname
        fig.savefig(figname)
        plt.close(fig)

    def optimize(self):
        """
        Optimize the objectives, i.e., discover the Pareto front.
        """
        self.ei_values = []
        for it in xrange(self.max_it):
            if self.verbose:
                print 'step {0:s}'.format(str(it).zfill(len(str(self.max_it))))
                #print '\t> training surrogates'
            #self.train_surrogates()
            # Are we drawing new design points or not?
            if isinstance(self.X_design, int):
                num_design = self.X_design
                X_design = design.latin_center(num_design, self.num_dim)   
            else:
                X_design = self.X_design
            if self.verbose:
                print '\t> done'
                print '\t> computing expected improvement'
            ei = self.compute_expected_improvement(X_design)
            if self.verbose:
                print '\t> done'
            i = np.argmax(ei)
            ei_max = ei[i]
            self.ei_values.append(ei_max)
            rel_ei_max = ei_max / self.ei_values[0]
            if self.verbose:
                print '\t> rel_ei_max = {0:1.3f}'.format(rel_ei_max)
            if it >= self.add_at_least and rel_ei_max < self.rtol:
                if self.verbose:
                    print '*** Converged (rel_ei_max = {0:1.7f} < rtol = {1:1.2e})'.format(rel_ei_max, self.rtol)
                    print '\t> writing final status'
                    self.plot_status(it,final=True)
                break
            if self.verbose:
                print '\t> adding design point', i
                print '\t> X_d[i, :]', X_design[i, :]
                print '\t> starting simulation'
            #print self.Y_pareto
            k = self.active_cells
            #for k in k:
                #print k
            lplus = self.active_cells_lplus
            #for lplus in lplus:
                #print lplus
            #y = self.obj_funcs(X_design[i,:])
            print "Run the experiment/code at the following design"+str(X_design[i,:])
            y = input('Enter the observed value at the new design')
            self.add_new_observations(X_design[i, :], y)
            if self.verbose:
                print '\t> training surrogates now'
            self.train_surrogates()
            self.Y_p = self.get_projected_observations()
            self.idx = get_idx_of_observed_pareto_front(self.Y_p)
            self.b = compute_sorted_list_of_pareto_points(self.Y_pareto, self.y_ref)
            #self.Y_true_noiseless = np.array([self.obj_funcs_true(x) for x in self.X])
            if self.verbose:
                print '\t> done'
            if not isinstance(self.X_design, int):
                self.X_design = np.delete(self.X_design, i, 0)
            if self.make_plots:
                if it==(self.max_it-1):
                    self.plot_status(it,final=True)
                else:
                    self.plot_status(it)

    def add_new_observations(self, x, y):
        """
        Add new observations and make sure all the quantities defined in
        __init__ are re-initialized.
        """
        self.X = np.vstack([self.X, x])
        self.Y = np.vstack([self.Y,[y]])
        mx = np.max(self.Y, axis=0)
        mn = np.min(self.Y, axis=0)
        self.y_ref = mx + 0.1 * (mx - mn)
        # The following can be updated fast (FIX)
        #self.Y_p = self.get_projected_observations()
        #self.idx = get_idx_of_observed_pareto_front(self.Y_p)
        #self.b = compute_sorted_list_of_pareto_points(self.Y_pareto, self.y_ref)

    def sample_pareto_fronts(self, num_of_design_samples=5,
                             num_of_gp=5,
                             num_of_design_points=1000, verbose=False):
        """
        Samples a plaussible pareto front.
        NOTE: Only works if design is the unit hyper-cube.
        """
        import  design
        Y_p = []
        for _ in xrange(num_of_design_samples):
            X_design = design.latin_center(num_of_design_points, self.X.shape[1])
            Y = []
            for m in self.surrogates:
                _m = copy.copy(m)
                _m.Gaussian_noise.variance.unconstrain()
                _m.Gaussian_noise.variance.fix(1e-8)
                y = _m.posterior_samples(X_design, size=num_of_gp, full_cov=True)
                Y.append(y)
            Y = np.array(Y)
            for i in xrange(Y.shape[2]):
                if verbose:
                    print 'sampling pareto', _, i
                idx = get_idx_of_observed_pareto_front(Y[:, :, i].T)
                y_p = Y[:, idx, i].T
                Y_p.append(y_p)
        return Y_p