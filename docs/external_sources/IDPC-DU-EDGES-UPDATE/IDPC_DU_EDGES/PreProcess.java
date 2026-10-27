package IDPC_DU.Multifactorial_Evolutionary_Algorithm.IDPC_DU_EDGES;

import java.util.ArrayList;
import java.util.List;

public class PreProcess {
    public static int CSS_NODES_NUM = 0;
    public static int [] css_node_out_edge;
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

    public static void EdgeFilter(List<Edge> edge) {
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
    }

    public static void CreateUSS(List<Edge> [] edges)
    {
        //Number of nodes in the new USS
        for (int i = 0; i < Files.NODES_NUM.length; i++) {
            if (CSS_NODES_NUM < Files.NODES_NUM[i]){
                CSS_NODES_NUM = Files.NODES_NUM[i];
            }
        }

        //Initialize number of nodes out for each edge and sample node
        css_node_out_edge = new int [CSS_NODES_NUM];
        sample_node = new int [CSS_NODES_NUM];

        //Initialize sample node
        for (int i = 0; i < sample_node.length; i++) {
            sample_node[i] = i + 1;
        }

        for (int i = 0; i < edges.length; i++) {
            PreProcess.CountOutEdge(Files.edges[i], Files.node_out_edge.get(i), Files.NODES_NUM[i]);
        }

        int [] nodePos = new int[Files.TASK_NUM];
        int counter = 0;
        for (int i = 0; i < css_node_out_edge.length; i++) {
            for (int j = 0; j < Files.TASK_NUM; j++) {
                if (counter < Files.node_out_edge.get(j).length) {
                    //Element j of nodePos is the counter-th element of j-th node out
                    nodePos[j] = Files.node_out_edge.get(j)[counter];
                } else  nodePos[j] = 0;
            }
            css_node_out_edge[i] = FindLargestEdge(nodePos);
            counter++;
        }
    }

    public static void SeperateData(List<Edge> edge, List<Edge> [][] edges_1) {
        for (int i = 0; i < edge.size(); i++) {
            int u = edge.get(i).src - 1;
            int v = edge.get(i).des - 1;
            edges_1[u][v].add(edge.get(i));
        }
    }

    public static int FindLargestEdge(int [] edges){
        int maxNode = 0;
        for (int i = 0; i < edges.length; i++) {
            if (maxNode <= edges[i]){
                maxNode = edges[i];
            } 
        }
        return maxNode;
    } 
}
