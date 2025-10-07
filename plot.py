import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

# ----------------- Plotting functions ----------------- #

def scatter_sol(ax, xyz, col, adj=1.05):
    # only plot the solutions where the y-coordinate is positive
    xyz = [adj*v for v in xyz if v[1] > 0]
    ax.scatter([v[0] for v in xyz], [v[1] for v in xyz], [v[2] for v in xyz], alpha=0.5, c=col)


def plot_sol(covs, vectors, S, objective_values, plot_eigs=True):
    # v_pca = v_pca.detach().numpy()
    dim = covs[0].shape[0]
    
    fig = plt.figure()
    if dim == 2:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(S[:, 0], S[:, 1], c=objective_values, cmap='viridis', s=100)
        fig.colorbar(scatter, ax=ax)
        # ax.scatter(v_pca[0], v_pca[1], c='r', s=300, label="minPCA solution")

    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(S[:, 0], S[:, 1], S[:, 2], c=objective_values, cmap='viridis', s=1, label="Objective value")
        # ax.scatter(v_pca[0], v_pca[1], v_pca[2], c='r', s=100)
        ax.set_zlabel('Z')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('minPCA objective values')
    
    # for each covariance matrix, plot the eigenvectors
    if plot_eigs:
        for cov in covs:
            eigs = np.linalg.eig(cov.detach().numpy())[1]
            for eig in eigs:
                # ax.quiver(0, 0, 0, eig[0], eig[1], eig[2], color='r')
                # plot the eigenvectors as a point
                if dim == 2:
                    ax.scatter(eig[0], eig[1], c='turquoise', s=200)
                if dim == 3: 
                    ax.scatter(eig[0], eig[1], eig[2], c='turquoise', s=10)

    # vectors is a dictionary, if non empty add
    markers = ['o', 'x', 's', 'D', 'v', '^', '<', '>', 'p', 'h']
    if vectors:
        for name, v in vectors.items():
            v = v.detach().numpy()
            if name == 'minpca': col = 'r'
            else: col = 'b'
            
            if dim == 2:
                ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=col)
                ax.scatter(v[0], v[1], c=col, s=100, label=name, marker=markers.pop())
            if dim == 3:
                ax.scatter(v[0], v[1], v[2], c=col, s=100, label=name, marker=markers.pop())
                
    plt.legend()            
    plt.show()


def plot_sol_interactive(covs, v_pca, S, f_vals):
    v_pca = v_pca.detach().numpy()

    # Create a scatter plot for the data points in S
    scatter_data = go.Scatter3d(
        x=S[:, 0, 0],
        y=S[:, 1, 0],
        z=S[:, 2, 0],
        mode='markers',
        marker=dict(
            size=3,
            color=f_vals,
            colorscale='Viridis',
            opacity=0.8
        ),
        name="Data Points"
    )

    # Create scatter plots for the eigenvectors of each covariance matrix
    eigenvectors = get_hull_eigenvectors(covs)
    colors = ['red', 'orange', 'wheat'] * 2
    eigenvector_points = []
    for i,eig in enumerate(eigenvectors):
        eigenvector_points.append(
            go.Scatter3d(
                x=[eig[0]],
                y=[eig[1]],
                z=[eig[2]],
                mode='markers',
                marker=dict(
                    size=10,
                    color=colors[i]
                ),
                name=f"Eigenvector Point {i%3}"
            )
        )

    if (angle(v_pca, eigenvectors[0]) > angle(-v_pca, eigenvectors[0])):
        v_pca = -v_pca
    
    # Create a scatter plot for the v_pca point
    pca_point = go.Scatter3d(
        x=[v_pca[0]],
        y=[v_pca[1]],
        z=[v_pca[2]],
        mode='markers',
        marker=dict(
            size=15,
            color='violet',
        ),
        name="PCA Point"
    )

    # Set up the layout
    layout = go.Layout(
        scene=dict(
            xaxis_title='X Label',
            yaxis_title='Y Label',
            zaxis_title='Z Label',
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    # Combine all traces and create the figure
    fig = go.Figure(data=[scatter_data, pca_point] + eigenvector_points, layout=layout)
    
    # Show the interactive plot
    fig.show()


# ----------------- Helper functions ----------------- #

def angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# here we get the eigenvectors of each of the covariance matrices
# then we find the combination (+/-) of the eigenvectors that have the smallest distance (angle) 
# to each other
def get_hull_eigenvectors(covs):
    print("WARNING: `get_hull_eigenvectors` is not working as intended.")
    hull_eigenvectors = []
    for cov in covs:
        eigs = np.linalg.eig(cov.detach().numpy())[1]
        hull_eigenvectors += [eigs[:, i] for i in range(eigs.shape[1])]
    
    for i in range(1, len(hull_eigenvectors)):
        print(i)
        if i == 1:
            print(hull_eigenvectors[i-1])
            angle1 = angle(hull_eigenvectors[i-1], hull_eigenvectors[i])
            angle2 = angle(hull_eigenvectors[i-1], -hull_eigenvectors[i])
            if angle1 > angle2:
                hull_eigenvectors[i] = -hull_eigenvectors[i]

        else:
            # find the biggest angle between the current eigenvector and any previous
            angles1 = []
            angles2 = []
            for j in range(i):
                angle1 = angle(hull_eigenvectors[i], hull_eigenvectors[j])
                angle2 = angle(-hull_eigenvectors[i], hull_eigenvectors[j])
                angles1.append(angle1)
                angles2.append(angle2)

            if np.max(angles1) > np.max(angles2):
                hull_eigenvectors[i] = -hull_eigenvectors[i]

    return hull_eigenvectors

# Generate evenly distributed points on the unit sphere using the Fibonacci lattice
def get_pts_on_sphere(n=1000, p=3):
    if p == 3:
        golden_ratio = (1 + np.sqrt(5)) / 2
        i = np.arange(n)
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * i / n)
        
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        # Return points as as array [[[x1], [y1], [z1]], [[x2], [y2], [z2]], ...]
        return np.array([x, y, z]).T.reshape(n, 3, 1)
    elif p == 2: 
        # Generate points on the unit circle
        theta = np.linspace(0, 2*np.pi, n)
        x = np.cos(theta)
        y = np.sin(theta)
        return np.array([x, y]).T.reshape(n, 2, 1)
