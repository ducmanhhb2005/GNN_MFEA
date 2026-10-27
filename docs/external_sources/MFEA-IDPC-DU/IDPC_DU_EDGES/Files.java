package IDPC_DU.Multifactorial_Evolutionary_Algorithm.IDPC_DU_EDGES;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class Files {
    public static int TASK_NUM = 2;
    public static int TASK1 = 0;
    public static int TASK2 = 1;

    public static int SRC1, SRC2;                               //source
    public static int DES1, DES2;                               //destination
    public static int NODES_NUM1, NODES_NUM2;                   //number of nodes
    public static int DOMAINS_NUM1, DOMAINS_NUM2;               //number of domains

    public static int [] node_out_edge1;
    public static int [] node_out_edge2;
    
    public static List<Edge> edges1 = new ArrayList<Edge>();    //All edges in task1
    public static List<Edge> edges2 = new ArrayList<Edge>();    //All edges in task2

    public static List<Edge> [][] edges_1;                      //Separate edges from each nodes in task1
    public static List<Edge> [][] edges_2;                      //Separate edges from each nodes in task2

    public static void ReadFile(){
        BufferedReader bReader;
        int counter = 0;

        //Read data for task1
        try  {
            bReader= new BufferedReader(new FileReader("C:/dataset/IDPC-DU/set1/set1/idpc_20x10x2492.idpc"));
            String line = bReader.readLine();

            while(line != null){
                if (counter == 0){
                    String [] splitedString= line.split("\\s+");
                    NODES_NUM1 = Integer.parseInt(splitedString[0]);
                    DOMAINS_NUM1 = Integer.parseInt(splitedString[1]);
                    edges_1  = new ArrayList[NODES_NUM1][NODES_NUM1];
                    counter++;
                    line = bReader.readLine();
                    // System.out.print(NODES_NUM1 + " ");
                    // System.out.print(DOMAINS_NUM1 + " ");
                    // System.out.println(" ");
                }
                else if (counter == 1){
                    String [] splitedString= line.split("\\s+");
                    SRC1 = Integer.parseInt(splitedString[0]);
                    DES1 = Integer.parseInt(splitedString[1]);
                    counter++;
                    line = bReader.readLine();
                    // System.out.print(SRC1 + " ");
                    // System.out.print(DES1 + " ");
                    // System.out.println(" ");
                }
                else {
                    int u, v, w, d;
                    String [] splitedString= line.split("\\s+");
                    u = Integer.parseInt(splitedString[0]);
                    //System.out.print(u + " ");
                    v = Integer.parseInt(splitedString[1]);
                    //System.out.print(v + " ");
                    w = Integer.parseInt(splitedString[2]);
                    //System.out.print(w + " ");
                    d = Integer.parseInt(splitedString[3]);
                    //System.out.print(d + " ");
                    //System.out.println(" ");
                    Edge edge = new Edge(u, v, w, d);
                    edges1.add(edge);
                    counter++;
                    line = bReader.readLine();
                }
            }
            bReader.close();
        } catch (Exception e)
        {
            e.printStackTrace();
        }

        //System.out.println("Task2");
        counter = 0;
        //Read data for task2
        try  {
            bReader= new BufferedReader(new FileReader("C:/dataset/IDPC-DU/set1/set1/idpc_45x22x43769.idpc"));
            String line = bReader.readLine();

            while(line != null){
                if (counter == 0){
                    String [] splitedString= line.split("\\s+");
                    NODES_NUM2 = Integer.parseInt(splitedString[0]);
                    DOMAINS_NUM2 = Integer.parseInt(splitedString[1]);
                    edges_2  = new ArrayList[NODES_NUM2][NODES_NUM2];
                    counter++;
                    line = bReader.readLine();
                    // System.out.print(NODES_NUM2 + " ");
                    // System.out.print(DOMAINS_NUM2 + " ");
                    // System.out.println(" ");
                }
                else if (counter == 1){
                    String [] splitedString= line.split("\\s+");
                    SRC2 = Integer.parseInt(splitedString[0]);
                    DES2 = Integer.parseInt(splitedString[1]);
                    counter++;
                    line = bReader.readLine();
                    // System.out.print(SRC2 + " ");
                    // System.out.print(DES2 + " ");
                    // System.out.println(" ");
                }
                else {
                    int u, v, w, d;
                    String [] splitedString= line.split("\\s+");
                    u = Integer.parseInt(splitedString[0]);
                    //System.out.print(u + " ");
                    v = Integer.parseInt(splitedString[1]);
                    //System.out.print(v + " ");
                    w = Integer.parseInt(splitedString[2]);
                    //System.out.print(w + " ");
                    d = Integer.parseInt(splitedString[3]);
                    // System.out.print(d + " ");
                    // System.out.println(" ");

                    Edge edge = new Edge(u, v, w, d);
                    edges2.add(edge);
                    counter++;
                    line = bReader.readLine();
                }
            }
            bReader.close();
        } catch (Exception e)
        {
            e.printStackTrace();
        }

        node_out_edge1 = new int [NODES_NUM1];
        node_out_edge2 = new int [NODES_NUM2];

        for (int i = 0; i < node_out_edge1.length; i++) {
            node_out_edge1[i] = 0;
        }
        for (int i = 0; i < node_out_edge2.length; i++) {
            node_out_edge2[i] = 0;
        }

        for (int i = 0; i < NODES_NUM1; i++) {
            for (int j = 0; j < NODES_NUM1; j++) {
                List<Edge> e = new ArrayList<Edge>();
                edges_1[i][j] = e;
            }
        }
        for (int i = 0; i < NODES_NUM2; i++) {
            for (int j = 0; j < NODES_NUM2; j++) {
                List<Edge> e = new ArrayList<Edge>();
                edges_2[i][j] = e;
            }
        }
    }
}
