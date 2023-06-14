import pickle
import numpy as np
import matplotlib.pyplot as plt

from ddplasma_utils import FiniteDiffPoint

def load_obj(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def get_sampling_points(ps_data, num_tux0, vol_tux, lowerbound_t = 200, upperbound_t = 300):

    np.random.seed(188)

    nt, nu, nx = ps_data['F'].shape

    # Sample a collection of data points, stay away from edges so I can just use centered finite differences.
    num_x0 = num_tux0[0]
    num_u0 = num_tux0[1]
    num_t0 = num_tux0[2]
    w_x  = vol_tux[0]
    w_u  = vol_tux[1]
    w_t  = vol_tux[2]
    num_vols = num_t0 * num_u0 * num_x0
    num_points_per_vol = (w_t*w_u*w_x)

    lowerbound_x = 10
    upperbound_x = nx-30
    lowerbound_u = 48
    upperbound_u = nu-48

    points = np.zeros((num_vols, 3, num_points_per_vol), dtype=int)

    f_thresh = 0.25

    tux_throws = 0
    while tux_throws < num_t0*num_u0*num_x0:
        x0 = np.random.randint(lowerbound_x, upperbound_x)
        u0 = np.random.randint(lowerbound_u, upperbound_u)
        t0 = np.random.choice(np.arange(lowerbound_t, upperbound_t), 1, replace=True)[0]
        if ps_data['F'][t0, u0 ,x0] > f_thresh:
            t_range, u_range, x_range = np.meshgrid(np.arange(t0, t0+w_t), np.arange(u0, u0+w_u), np.arange(x0, x0+w_x), indexing='xy')
            points[tux_throws, 0, :] = t_range.flatten()
            points[tux_throws, 1, :] = u_range.flatten()
            points[tux_throws, 2, :] = x_range.flatten()
            tux_throws +=1

    return points

def plot_sample_points_on_data(ps_data, fld_data, points):
    ps = []
    for i in range(len(points)):
        t, u, x = points[i]
        for j in range(len(t)):
            ps.append([x[j], u[j], t[j]])
    ps = np.array(ps)

    i_t = 320

    x_arr = ps_data['x']
    u_arr = ps_data['u']
    t_arr = ps_data['t']

    dx = x_arr[1] - x_arr[0]
    du = u_arr[1] - u_arr[0]
    dt = t_arr[1] - t_arr[0]

    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.title(r'$F(x,u,t ='+ str(round(t_arr[i_t],2))+'/\omega_p)$')
    plt.imshow(ps_data['F'][i_t, :, :], origin='lower',
               extent=[x_arr[0], x_arr[-1], u_arr[0], u_arr[-1]], aspect='auto', cmap = 'gist_heat_r', vmax = 7)
    plt.xlabel(r'$x\omega_{pe}/c$')
    plt.ylabel(r'$u/c$')
    plt.colorbar(label = r'$f$')
    plt.scatter(ps[:,0]*dx, u_arr[0] + ps[:,1]*du, c = 'k', s=0.25)

    plt.subplot(1,2,2)
    plt.title(r'$E1$')
    plt.imshow(fld_data['E'][:, :], origin='lower',
               extent=[x_arr[0], x_arr[-1], t_arr[0], t_arr[-1]], aspect='auto', cmap = 'RdBu')
    plt.xlabel(r'$x\omega_{pe}/c$')
    plt.ylabel(r'$t\omega_{pe}$')
    plt.colorbar(label = r'$E$')
    plt.scatter(x_arr[0] + ps[:,0]*dx, t_arr[0] + ps[:,2]*dt, c = 'k', s=0.25)
    plt.tight_layout()
    plt.show()

def collect_sample_data_and_derivatives(ps_data, fld_data, points, O = 2):
    # This routine evaluates the primary candidate PDE terms at sampled points on the data.
    # By primary candidate PDE terms I mean the main dynamical variables of the system and their gradients.
    # In this example, the main dynamical variables are the distribution function (F), the electric field (E) and the phase space coordinates (x,v).
    # This routine outputs the values of (F, E, x, v, and their gradients) at the sample points in a dictionary called "features".
    # The time derivative of F (F_t), which is the target variable for which we want to discover the governing PDE, is output in the dictionary "y_quants".
    # The "O" parameter controls the order of accuracy of the finite differencing stencils to estimate derivatives.
    # By default we use "O = 2" corresponding to second order accurate finite differences.

    num_vols = len(points)
    num_points_per_vol = np.array(points[1]).shape[-1]

    dx = ps_data['dx']
    du = ps_data['du']
    dt = ps_data['dt']

    # initialize y_quants
    y_quants = {}
    for quant in ['F_t']:
        y_quants[quant] = np.zeros((num_vols, num_points_per_vol, 1))

    # initialize features dictionary
    features = {}
    ps_quants = ['F', 'v', 'u', 'x']
    fld_quants = ['E']
    for quant in ps_quants+fld_quants:
        features[quant] = np.zeros((num_vols, num_points_per_vol, 1))
        if quant not in ['v', 'u', 'x']:
            features[quant+'_x']  = np.zeros((num_vols, num_points_per_vol, 1))
            features[quant+'_u']  = np.zeros((num_vols, num_points_per_vol, 1))
            features[quant+'_xx'] = np.zeros((num_vols, num_points_per_vol, 1))
            features[quant+'_uu'] = np.zeros((num_vols, num_points_per_vol, 1))


    # Evaluating derivatives
    for p in range(len(points)):
        if p%int(len(points)/10) ==0:
            print('Progress: ', np.round(p/len(points)*100, -1), '%')

        [t_range, u_range, x_range]  = points[p]

        for i in range(num_points_per_vol):
            t, u, x = t_range[i], u_range[i], x_range[i]

            # Species data
            for quant in ['F']:
                features[quant][p, i] = ps_data[quant][t, u, x]
                features[quant+'_x'][p, i]  = FiniteDiffPoint(ps_data[quant][t, u, :], i_x = x,
                                                              dx = dx, d = 1, O = O)
                features[quant+'_u'][p, i]  = FiniteDiffPoint(ps_data[quant][t, :, x], i_x = u,
                                                              dx = du, d = 1, O = O)
                features[quant+'_xx'][p, i] = FiniteDiffPoint(ps_data[quant][t, u, :], i_x = x,
                                                              dx = dx, d = 2, O = O)
                features[quant+'_uu'][p, i] = FiniteDiffPoint(ps_data[quant][t, :, x], i_x = u,
                                                              dx = du, d = 2, O = O)

            features['v'][p, i] = ps_data['v'][u]
            features['u'][p, i] = ps_data['u'][u]
            features['x'][p, i] = ps_data['x'][x]

            # Fld data
            for quant in fld_quants:
                features[quant][p, i] = fld_data[quant][t, x]
                features[quant+'_x'][p, i]  = FiniteDiffPoint(fld_data[quant][t, :], i_x = x,
                                                              dx = dx, d = 1, O = O)
                features[quant+'_xx'][p, i] = FiniteDiffPoint(fld_data[quant][t, :], i_x = x,
                                                              dx = dx, d = 2, O = O)

            # Evaluate time derivative of F
            y_quants['F_t'][p, i] = FiniteDiffPoint(ps_data['F'][:, u, x], i_x = t, dx = dt, d = 1, O = O)

    return features, y_quants
