import pandas as pd
import plotly.express as px
from dash import Input, Output, dcc, html
from jupyter_dash import JupyterDash
from scipy.stats.mstats import trimmed_var
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
JupyterDash.infer_jupyter_proxy_config()
import numpy as np
import plotly.graph_objects as go

def wrangle(filepath):

    """Read SCF data file into ``DataFrame``.

    Returns only credit fearful households whose net worth is less than $2 million.

    Parameters
    ----------
    filepath : str
        Location of CSV file.
    """
    df=pd.read_csv(filepath)
    
    #create mask
    mask=(df["TURNFEAR"]==1)&(df["NETWORTH"]<2e6)
    
    df=df[mask]
    return df


df = wrangle("data/SCFP2019.csv.gz")



app = JupyterDash(__name__)



def get_high_var_features(trimmed=True,return_feat_names=True):

    """Returns the five highest-variance features of ``df``.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    return_feat_names : bool, default=False
        If ``True``, returns feature names as a ``list``. If ``False``
        returns ``Series``, where index is feature names and values are
        variances.
    """
    #calculate variance
    if trimmed:
        top_five_features=(
            df.apply(trimmed_var).sort_values().tail(5)
        )
    else:
        top_five_features=df.var().sort_values().tail(5)
#     Extract names
    if return_feat_names:
        top_five_features=top_five_features.index.tolist()
    return top_five_features



@app.callback(
    Output("bar-chart","figure"),Input("trim-button","value")
)
def serve_bar_chart(trimmed=True):

    """Returns a horizontal bar chart of five highest-variance features.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.
    """
    # get features
    top_five_features=get_high_var_features(trimmed=trimmed,return_feat_names=False)
    
    #build bar chart
    fig=px.bar(x=top_five_features,y=top_five_features.index,orientation="h")
    fig.update_layout(xaxis_title="Variance",yaxis_title="Features")
    return fig



def get_model_metrics(trimmed=True,k=2,return_metrics=False):

    """Build ``KMeans`` model based on five highest-variance features in ``df``.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    k : int, default=2
        Number of clusters.

    return_metrics : bool, default=False
        If ``False`` returns ``KMeans`` model. If ``True`` returns ``dict``
        with inertia and silhouette score.

    """
    #get high var features
    features=get_high_var_features(trimmed=trimmed,return_feat_names=True)
    X=df[features]
    model=make_pipeline(StandardScaler(),KMeans(n_clusters=k,random_state=42))
    model.fit(X)
    if return_metrics:
        #inertia
        i=model.named_steps["kmeans"].inertia_
        #ss
        ss=silhouette_score(X,model.named_steps["kmeans"].labels_)
        metrics={
            "inertia":round(i),
            "silhouette":round(ss,3)
        }
        return metrics
    return model


@app.callback(
    Output("metrics","children"),
    Input("trim-button","value"),
    Input("k-slider","value")
)
def serve_metrics(trimmed=True,k=2):

    """Returns list of ``H3`` elements containing inertia and silhouette score
    for ``KMeans`` model.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    k : int, default=2
        Number of clusters.
    """
    #get metrics
    metrics=get_model_metrics(trimmed=trimmed,k=k,return_metrics=True)
    
    #add metrics to html elements
    text=[
        html.H3(f"Inertia: {metrics['inertia']}"),
        html.H3(f"Silhouette Score: {metrics['silhouette']}")
    ]
    
    
    return text


def get_pca_labels(trimmed=True,k=2):

    """
    ``KMeans`` labels.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    k : int, default=2
        Number of clusters.
    """
    #create feature matrix
    features = get_high_var_features(trimmed=trimmed,return_feat_names=True)
    X=df[features]
    
    #build transformer
    transformer = PCA(n_components=2,random_state=42)
    X_t=transformer.fit_transform(X)
    
    X_pca=pd.DataFrame(X_t,columns=["PC1","PC2"])
    
    #add labels
    model=get_model_metrics(trimmed=trimmed,k=k,return_metrics=False)
    X_pca["labels"]=model.named_steps["kmeans"].labels_.astype(str)
    X_pca.sort_values("labels",inplace=True)
    return X_pca


def get_pca_labels_and_centers(trimmed=True, k=2):
    features = get_high_var_features(trimmed=trimmed, return_feat_names=True)
    X = df[features]
    transformer = PCA(n_components=2, random_state=42)
    X_t = transformer.fit_transform(X)
    X_pca = pd.DataFrame(X_t, columns=["PC1", "PC2"])
    model = make_pipeline(StandardScaler(), KMeans(n_clusters=k, random_state=42))
    model.fit(X)
    X_pca["labels"] = model.named_steps["kmeans"].labels_.astype(str)
    X_pca.sort_values("labels", inplace=True)
    # Get cluster centers in PCA space
    centers = model.named_steps["kmeans"].cluster_centers_
    centers_scaled = model.named_steps["standardscaler"].inverse_transform(centers)
    centers_pca = transformer.transform(centers_scaled)
    centers_df = pd.DataFrame(centers_pca, columns=["PC1", "PC2"])
    centers_df["labels"] = [str(i) for i in range(k)]
    return X_pca, centers_df


@app.callback(
    Output("pca-scatter", "figure"),
    Input("trim-button", "value"),
    Input("k-slider", "value"),
    Input("centroid-toggle", "value")
)
def serve_scatter_plot(trimmed=True, k=2, centroid_toggle=[]):
    X_pca, centers_df = get_pca_labels_and_centers(trimmed=trimmed, k=k)
    fig = px.scatter(
        data_frame=X_pca,
        x="PC1",
        y="PC2",
        color="labels",
        title="PCA Representation of clusters"
    )
    if "show" in centroid_toggle:
        # Add ellipses for each cluster
        for label in X_pca["labels"].unique():
            cluster = X_pca[X_pca["labels"] == label]
            x_ellipse, y_ellipse = get_ellipse(cluster["PC1"], cluster["PC2"])
            fig.add_trace(
                go.Scatter(
                    x=x_ellipse,
                    y=y_ellipse,
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0.5)", dash="dot"),
                    name=f"Ellipse {label}",
                    showlegend=False
                )
            )
        # Add centroids
        fig.add_scatter(
            x=centers_df["PC1"],
            y=centers_df["PC2"],
            mode="markers",
            marker=dict(symbol="x", size=15, color="black"),
            name="Cluster Centers"
        )
    fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
    return fig


print("app type:", type(app))


app.layout = html.Div(
    [
        #APPLICATION TITLE
        html.H1("Survey of Consumer Finances"),
        #Bar chart element
        html.H2("High Variance Features"),
        #Graph
        dcc.Graph(figure=serve_bar_chart(),id="bar-chart"),
        
        dcc.RadioItems(
            options=[
                {"label":"trimmed","value":True},{"label":"not-trimmed","value":False}
            ],
            value=True,
            id="trim-button"
        ),
        html.H2("K-means Clustering"),
        html.H3("Number of Clusters (k)"),
        dcc.Slider(min=2,max=12,step=2,value=2,id="k-slider"),
        html.Div(id="metrics"),
        dcc.Graph(id="pca-scatter"),
        html.H3("Show Cluster Centroids"),
        dcc.Checklist(
            options=[{"label": "Show Centroids & Ellipses", "value": "show"}],
            value=[],
            id="centroid-toggle",
            inline=True
        ),
        html.H2("Cluster Summary"),
        html.Div(id="summary"),
        html.H2("K-means Metrics"),
        dcc.Graph(id="k-metrics-plot")
    ]
)

def get_ellipse(x, y, n_std=2.0, num_points=100):
    """
    Returns the x and y coordinates of an ellipse enclosing the cluster.
    n_std: Number of standard deviations (default 2 for ~95% coverage)
    """
    if len(x) < 2:
        return x, y  # Not enough points for ellipse
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    t = np.linspace(0, 2*np.pi, num_points)
    ellipse = np.array([width/2 * np.cos(t), height/2 * np.sin(t)])
    R = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                  [np.sin(np.radians(theta)),  np.cos(np.radians(theta))]])
    ellipse_rot = R @ ellipse
    x_ellipse = ellipse_rot[0] + np.mean(x)
    y_ellipse = ellipse_rot[1] + np.mean(y)
    return x_ellipse, y_ellipse

def get_cluster_summary(trimmed=True, k=2):
    X_pca, centers_df = get_pca_labels_and_centers(trimmed=trimmed, k=k)
    summary = []
    # Cluster sizes
    cluster_counts = X_pca["labels"].value_counts().sort_index()
    summary.append(html.H4("Cluster Sizes:"))
    summary.append(html.Ul([html.Li(f"Cluster {label}: {count} points") for label, count in cluster_counts.items()]))
    # Cluster centers
    summary.append(html.H4("Cluster Centers (PCA space):"))
    summary.append(
        html.Table([
            html.Tr([html.Th("Cluster"), html.Th("PC1"), html.Th("PC2")])
        ] + [
            html.Tr([html.Td(label), html.Td(f"{row.PC1:.2f}"), html.Td(f"{row.PC2:.2f}")])
            for label, row in centers_df.iterrows()
        ])
    )
    return summary

@app.callback(
    Output("summary", "children"),
    Input("trim-button", "value"),
    Input("k-slider", "value")
)
def serve_summary(trimmed=True, k=2):
    return get_cluster_summary(trimmed=trimmed, k=k)

def get_k_metrics(trimmed=True, k_range=range(2, 13, 2)):
    inertias = []
    silhouettes = []
    for k in k_range:
        features = get_high_var_features(trimmed=trimmed, return_feat_names=True)
        X = df[features]
        model = make_pipeline(StandardScaler(), KMeans(n_clusters=k, random_state=42))
        model.fit(X)
        inertia = model.named_steps["kmeans"].inertia_
        # Silhouette score only makes sense if k < n_samples
        if X.shape[0] > k:
            sil = silhouette_score(X, model.named_steps["kmeans"].labels_)
        else:
            sil = np.nan
        inertias.append(inertia)
        silhouettes.append(sil)
    return list(k_range), inertias, silhouettes

@app.callback(
    Output("k-metrics-plot", "figure"),
    Input("trim-button", "value")
)
def serve_k_metrics_plot(trimmed=True):
    k_range, inertias, silhouettes = get_k_metrics(trimmed=trimmed)
    fig = go.Figure()
    # Inertia (primary y-axis)
    fig.add_trace(go.Scatter(
        x=k_range, y=inertias, mode="lines+markers", name="Inertia", yaxis="y1"
    ))
    # Silhouette Score (secondary y-axis)
    fig.add_trace(go.Scatter(
        x=k_range, y=silhouettes, mode="lines+markers", name="Silhouette Score", yaxis="y2"
    ))
    fig.update_layout(
        title="Inertia and Silhouette Score vs. Number of Clusters (k)",
        xaxis=dict(title="Number of Clusters (k)"),
        yaxis=dict(title="Inertia"),
        yaxis2=dict(title="Silhouette Score", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99)
    )
    return fig


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=False)
