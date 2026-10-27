package IDPC_DU.Multifactorial_Evolutionary_Algorithm.IDPC_DU_EDGES;

import java.util.ArrayList;
import java.util.List;

public class PreProcess {
    public static int NODES_NUM;
    public static int [] node_out_edge;
    public static int [] sample_node;
    public static List<Edge> USS = new ArrayList<Edge>();

    public static void CountOutEdge(List<Edge> edge, int [] edgeOut ,int totalNode) {
        int count = 1;

        while (count <= totalNode){
            for (int i = 0; i < edge.size(); i++) {
                if(edge.get(i).src == count) {
                    edgeOut[count-1]++;
                }
            }
            count++;
        }
    }

    public static List<Edge> EdgeFilter(List<Edge> edge) {
        int i = 0, j = 0;
        boolean isBreak = false;

        while(i < edge.size()) {
            while(j < edge.size()) {
                if (i == j) {
                    j++;
                    continue;
                }
                else if((edge.get(i).src == edge.get(j).src) && (edge.get(i).des == edge.get(j).des) 
                && (edge.get(i).domain == edge.get(j).domain)) {
                    if(edge.get(i).weight <= edge.get(j).weight) {
                        // System.out.print("src i " + edge.get(i).src + " des " + edge.get(i).des
                        //  + " weight " + edge.get(i).weight + " domain " + edge.get(i).domain + 
                        //  " src j " + edge.get(j).src + " des " + edge.get(j).des
                        //  + " weight " + edge.get(j).weight + " domain " + edge.get(j).domain);
                        //  System.out.println(" ");
                        edge.remove(edge.get(j));
                        continue;
                    } else {
                        edge.remove(edge.get(i));
                        isBreak = true;
                        break;
                    }
                }
                j++;
            }

            if(isBreak){
                isBreak = false;
                continue;
            } else {
                i++;
                j = 0;
            }
        } 
        return edge;
    }

    public static void CreateUSS(List<Edge> edges1, List<Edge> edges2)
    {
        //Number of nodes and domains in the new USS
        if(Files.NODES_NUM1 < Files.NODES_NUM2){
            NODES_NUM = Files.NODES_NUM2;
        }
        else NODES_NUM = Files.NODES_NUM1;

        node_out_edge = new int [NODES_NUM];
        sample_node = new int [NODES_NUM];

        //Initialize sample node
        for (int i = 0; i < sample_node.length; i++) {
            sample_node[i] = i + 1;
        }

        PreProcess.CountOutEdge(Files.edges1, Files.node_out_edge1, Files.NODES_NUM1);
        PreProcess.CountOutEdge(Files.edges2, Files.node_out_edge2, Files.NODES_NUM2);

        if (Files.NODES_NUM1 < Files.NODES_NUM2){
            for (int i = 0; i < Files.node_out_edge1.length; i++) {
                if (Files.node_out_edge1[i] <= Files.node_out_edge2[i]){
                    node_out_edge[i] = Files.node_out_edge2[i];
                } else node_out_edge[i] = Files.node_out_edge1[i];
            }
            for (int i = Files.node_out_edge1.length; i < Files.node_out_edge2.length; i++) {
                node_out_edge[i] = Files.node_out_edge2[i];
            }
        } else if (Files.NODES_NUM1 == Files.NODES_NUM2){
            for (int i = 0; i < Files.node_out_edge1.length; i++) {
                if (Files.node_out_edge1[i] <= Files.node_out_edge2[i]){
                    node_out_edge[i] = Files.node_out_edge2[i];
                } else node_out_edge[i] = Files.node_out_edge1[i];
            }
        } else {
            for (int i = 0; i < Files.node_out_edge2.length; i++) {
                if (Files.node_out_edge1[i] <= Files.node_out_edge2[i]){
                    node_out_edge[i] = Files.node_out_edge2[i];
                } else node_out_edge[i] = Files.node_out_edge1[i];
            }
            for (int i = Files.node_out_edge2.length; i < Files.node_out_edge1.length; i++) {
                node_out_edge[i] = Files.node_out_edge1[i];
            }
        }
    }

    public static void SeperateData(List<Edge> edge, List<Edge> [][] edges_1) {
        for (int i = 0; i < edge.size(); i++) {
            int u = edge.get(i).src - 1;
            int v = edge.get(i).des - 1;
            edges_1[u][v].add(edge.get(i));
        }

        // for (int i = 0; i < Files.NODES_NUM1; i++) {
        //    for (int j = 0; j < Files.NODES_NUM1; j++) {
        //        for (int j2 = 0; j2 < edges_1[i][j].size(); j2++) {
        //         System.out.print("src " + edges_1[i][j].get(j2).src + " des " + edges_1[i][j].get(j2).des
        //         + " weight " + edges_1[i][j].get(j2).weight + " domain " + edges_1[i][j].get(j2).domain);
        //         System.out.println(" ");
        //        }
        //    } 
        // }
    }

}
