package IDPC_DU.Multifactorial_Evolutionary_Algorithm.IDPC_DU_EDGES;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class Files {
    public static int TASK_NUM;
    public static int [] TASK;

    public static String path = "C:/dataset/IDPC-DU/set1/set1/";
    public static String [] DATASET = {"idpc_10x5x425.idpc" , "idpc_10x10x1000.idpc", "idpc_10x20x2713.idpc", "idpc_15x15x3375.idpc", "idpc_20x10x2492.idpc", "idpc_45x45x91125.idpc"};

    public static int [] SRC;                                     //source
    public static int [] DES;                                     //destination
    public static int [] NODES_NUM;                               //number of nodes
    public static int [] DOMAINS_NUM;                             //number of domains
    //number of nodes out in each edges
    public static List<int[]> node_out_edge = new ArrayList<int[]>(); 
    public static List<Edge> [] edges;                            //Store all edges in each tasks
    //Separate edges from each nodes in each task
    public static List<List<Edge> [][]> edgesTask  = new ArrayList<List<Edge> [][]>();

    public static void CreateData(int taskNum){
        TASK_NUM = taskNum;
        TASK = new int[taskNum];
        for (int i = 0; i < TASK.length; i++) {
            TASK[i] = i;
        }
        NODES_NUM = new int[taskNum];
        DOMAINS_NUM = new int[taskNum];
        SRC = new int[taskNum];
        DES = new int[taskNum];

        edges = new ArrayList[taskNum];
        for (int i = 0; i < edges.length; i++) {
            edges[i] = new ArrayList<Edge>();
        }
    }

    //Read data from available file
    public static void ReadFile(int taskNum){
        BufferedReader bReader;
        int counter = 0;

        for (int i = 0; i < taskNum; i++) {
            counter = 0;
            try  {
                bReader= new BufferedReader(new FileReader(path + DATASET[i]));
                String line = bReader.readLine();
    
                while(line != null){
                    if (counter == 0){
                        String [] splitedString= line.split("\\s+");
                        NODES_NUM[i] = Integer.parseInt(splitedString[0]);
                        DOMAINS_NUM[i] = Integer.parseInt(splitedString[1]);
                        edgesTask.add(new ArrayList [NODES_NUM[i]][NODES_NUM[i]]);
                        counter++;
                        line = bReader.readLine();
                    }
                    else if (counter == 1){
                        String [] splitedString= line.split("\\s+");
                        SRC[i] = Integer.parseInt(splitedString[0]);
                        DES[i] = Integer.parseInt(splitedString[1]);
                        counter++;
                        line = bReader.readLine();
                    }
                    else {
                        int u, v, w, d;
                        String [] splitedString= line.split("\\s+");
                        u = Integer.parseInt(splitedString[0]);
                        v = Integer.parseInt(splitedString[1]);
                        w = Integer.parseInt(splitedString[2]);
                        d = Integer.parseInt(splitedString[3]);
                        Edge edge = new Edge(u, v, w, d);
                        edges[i].add(edge);
                        counter++;
                        line = bReader.readLine();
                    }
                }
                bReader.close();
            } catch (Exception e)
            {
                e.printStackTrace();
            }

            //Initialize the number of node out from each edge
            node_out_edge.add(new int [NODES_NUM[i]]);
            for (int j = 0; j < node_out_edge.get(i).length; j++) {
                node_out_edge.get(i)[j] = 0;
            }

            for (int j = 0; j < NODES_NUM[i]; j++) {
                for (int k = 0; k < NODES_NUM[i]; k++) {
                    edgesTask.get(i)[j][k] = new ArrayList<Edge>();
                }
            }
        }
    }
}