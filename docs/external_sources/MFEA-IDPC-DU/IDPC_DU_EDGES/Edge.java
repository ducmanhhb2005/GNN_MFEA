package IDPC_DU.Multifactorial_Evolutionary_Algorithm.IDPC_DU_EDGES;

public class Edge {
    public int src;
    public int des;
    public int weight;
    public int domain;

    public Edge(int u, int v, int w, int d){
        this.src = u;
        this.des = v;
        this.weight = w;
        this.domain = d;
    }
}
