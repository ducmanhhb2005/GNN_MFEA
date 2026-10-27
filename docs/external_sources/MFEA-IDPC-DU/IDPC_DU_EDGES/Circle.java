package IDPC_DU.Multifactorial_Evolutionary_Algorithm.IDPC_DU_EDGES;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class Circle {
    public int [] nodePri;      //circle
    public int [] edgeIdx;      
    public List<Integer> domainLeft = new ArrayList<Integer>();    
    public List<Integer> nodeLeft = new ArrayList<Integer>(); 

    public int [] f_cost = new int[Files.TASK_NUM];
    public int [] f_rank = new int[Files.TASK_NUM];
    public double[] s_fit = new double[Files.TASK_NUM];
    public double best_fit;
    public int s_factor;

    //Use for child only
    public Circle(Circle c1){
        this.nodePri = new int[c1.nodePri.length];
        this.edgeIdx = new int[c1.edgeIdx.length];

        //Assigning edge index 
        for (int i = 0; i < c1.edgeIdx.length; i++) {
            this.edgeIdx[i] = c1.edgeIdx[i];
        }
    }

    public Circle(int [] sampleNode, int [] nodeOut){
        int pos, tmp;
        this.nodePri = new int[sampleNode.length];
        this.edgeIdx = new int[sampleNode.length];

        for (int i = 0; i < sampleNode.length; i++) {
            nodePri[i] = sampleNode[i];
        }

        //Set the priority for each edge
        for (int i = 0; i < nodePri.length; i++) {
            pos = Main.random.nextInt(nodePri.length - 1);
            tmp = nodePri[i];
            nodePri[i] = nodePri[pos];
            nodePri[pos] = tmp;
        }

        //Set the edge-out index for each edge
        for (int i = 0; i < edgeIdx.length; i++) {
            if (nodeOut[i] == 0) {
                edgeIdx[i] = -1;
            } else {
                pos = Main.random.nextInt(nodeOut[i]);
                edgeIdx[i] = nodeOut[i];
            }
        }
    }

    public int getFirstTaskFcost() {
        return f_cost[Files.TASK1];
    }

    public int getSecTaskFcost() {
        return f_cost[Files.TASK2];
    }

    public double getBestFit() {
        return best_fit;
    }

    public int CalculateFactorialCost(Circle c, boolean task1) {
        int cost = 0, maxPri = 0, des, nextNode = -1, isVisited;
        List<Integer> adjacentNode = new ArrayList<Integer>();

        this.domainLeft.clear();
        this.nodeLeft.clear();

        //Task 1
        if (task1){
            isVisited = Files.SRC1;
            des = Files.DES1;
            c.nodeLeft.add(isVisited);
            int [] tmpNP = new int [Files.NODES_NUM1];
            int [] tmpEI = new int [Files.NODES_NUM1];

            if (Files.NODES_NUM1 < c.nodePri.length){
                tmpNP = DecodeNodePri(c.nodePri, Files.NODES_NUM1);
                tmpEI = DecodeEdgeIdx(c.edgeIdx, Files.NODES_NUM1);
            } else {
                tmpNP = c.nodePri;
                tmpEI = c.edgeIdx;
            }

            while (isVisited != des) {
                //Check which node is the adjacent one
                for (int j = 0; j < Files.NODES_NUM1; j++) {
                    for (int i = 0; i < Files.edges_1[isVisited-1][j].size(); i++) {
                        if (Files.edges_1[isVisited-1][j].get(i).src == isVisited
                        && !adjacentNode.contains(Files.edges_1[isVisited-1][j].get(i).des)) {
                            adjacentNode.add(Files.edges_1[isVisited-1][j].get(i).des);
                        }
                    }   
                }

                //Check which one got the highest priority based on adjacent nodes
                for (int i = 0; i < adjacentNode.size(); i++) {
                    if (maxPri < tmpNP[adjacentNode.get(i)-1] && !c.nodeLeft.contains(adjacentNode.get(i))) {
                        maxPri = tmpNP[adjacentNode.get(i)-1];
                        nextNode = adjacentNode.get(i);
                    }
                }
                
                //Calculate the factorial cost
                int penalty = 9999;
                if (adjacentNode.size() == 0 || c.domainLeft.contains(Files.edges_1[isVisited-1][nextNode-1].get(c.edgeIdx[isVisited-1] % Files.edges_1[isVisited-1][nextNode-1].size()).domain)) {
                    cost += penalty;
                } else {
                    cost += Files.edges_1[isVisited-1][nextNode-1].get(tmpEI[isVisited-1]  % Files.edges_1[isVisited-1][nextNode-1].size()).weight;
                    c.domainLeft.add(Files.edges_1[isVisited-1][nextNode-1].get(tmpEI[isVisited-1] % Files.edges_1[isVisited-1][nextNode-1].size()).domain);
                }
                // System.out.println("next node " + nextNode);
                // System.out.println("path num " + c.edgeIdx[isVisited-1]  % Files.edges_1[isVisited-1][nextNode-1].size());
                // System.out.println("cost " + cost);
                isVisited = nextNode;
                c.nodeLeft.add(isVisited);
                maxPri = 0;
                adjacentNode.clear();
            }
            return cost;
        } 
        //Task 2
        else {
            isVisited = Files.SRC2;
            des = Files.DES2;
            c.nodeLeft.add(isVisited);

            int [] tmpNP = new int [Files.NODES_NUM2];
            int [] tmpEI = new int [Files.NODES_NUM2];

            if (Files.NODES_NUM2 < c.nodePri.length){
                tmpNP = DecodeNodePri(c.nodePri, Files.NODES_NUM2);
                tmpEI = DecodeEdgeIdx(c.edgeIdx, Files.NODES_NUM2);
            } else {
                tmpNP = c.nodePri;
                tmpEI = c.edgeIdx;
            }

            while (isVisited != des) {
                //Check which node is the adjacent one 
                for (int j = 0; j < Files.NODES_NUM2; j++) {
                    for (int i = 0; i < Files.edges_2[isVisited-1][j].size(); i++) {
                        if (Files.edges_2[isVisited-1][j].get(i).src == isVisited
                        && !adjacentNode.contains(Files.edges_2[isVisited-1][j].get(i).des)) {
                            adjacentNode.add(Files.edges_2[isVisited-1][j].get(i).des);
                        }
                    }
                }

                //Check which one got the highest priority based on adjacent nodes for task 2
                for (int i = 0; i < adjacentNode.size(); i++) {
                    if (maxPri < tmpNP[adjacentNode.get(i)-1] && !c.nodeLeft.contains(adjacentNode.get(i))) {
                        maxPri = tmpNP[adjacentNode.get(i)-1];
                        nextNode = adjacentNode.get(i);
                    }
                }
                //Calculate the factorial cost for task 2
                int penalty = 9999;
                if (adjacentNode.size() == 0 || c.domainLeft.contains(Files.edges_2[isVisited-1][nextNode-1].get(tmpEI[isVisited-1] % Files.edges_2[isVisited-1][nextNode-1].size()).domain)) 
                {
                    cost += penalty;
                } else {
                    cost += Files.edges_2[isVisited-1][nextNode-1].get(tmpEI[isVisited-1] % Files.edges_2[isVisited-1][nextNode-1].size()).weight;
                    c.domainLeft.add(Files.edges_2[isVisited-1][nextNode-1].get(tmpEI[isVisited-1] % Files.edges_2[isVisited-1][nextNode-1].size()).domain);
                }
                isVisited = nextNode;
                c.nodeLeft.add(isVisited);
                maxPri = 0;
                adjacentNode.clear();
            }
        return cost;
        }
    }
    
    //Decode for each task
    public static int [] DecodeNodePri(int[] np, int nodeNum){
        int [] nodePri = new int [nodeNum]; 
        int counter = 0;
        for (int i = 0; i < np.length; i++) {
            if (np[i] <= nodeNum){
                nodePri[counter] = np[i];
                counter++;
            }
        }
        return nodePri;
    }

    public static int [] DecodeEdgeIdx(int[] ei, int nodeNum){
        int [] edgeIdx = new int [nodeNum]; 
        for (int i = 0; i < edgeIdx.length; i++) {
            edgeIdx[i] = ei[i];
        }
        return edgeIdx;
    }

    public static Comparator<Circle> FirstTaskCostComp = new Comparator<Circle>(){
        public int compare(Circle c1, Circle c2) {
            int cost1 = c1.getFirstTaskFcost();
            int cost2 = c2.getFirstTaskFcost();

            return cost1 - cost2;
        }
    };

    public static Comparator<Circle> SecTaskCostComp = new Comparator<Circle>(){
        public int compare(Circle c1, Circle c2) {
            int cost1 = c1.getSecTaskFcost();
            int cost2 = c2.getSecTaskFcost();

            return cost1 - cost2;
        }
    };

    public static Comparator<Circle> BestFitComp = new Comparator<Circle>(){
        public int compare(Circle c1, Circle c2) {
            double fit1 = c1.getBestFit();
            double fit2 = c2.getBestFit();
            double res =  fit1 - fit2;
            if(res > 0.00001) return -1;
            if(res < -0.00001) return 1;
            return 0;
        }
    };

    public static void ScalarFitnessCalculation(Circle c) {
        for (int i = 0; i < c.f_rank.length; i++) {
            c.s_fit[i] = (double) 1/c.f_rank[i];
        }
    }

    public static double BestFitEvaluation(Circle c) {
        double bf = c.s_fit[0];
        for (int i = 0; i < c.f_rank.length; i++) {
            if (c.s_fit[i] > bf) {
                bf = c.s_fit[i];
            }
        }
        return bf;
    }

    //Gan Skill Factor
    public static int SkillFactorAssigning(double[] sf){
        int sfactor = 0;
        double max = sf[0];
        for (int i = 0; i < sf.length; i++) {
            if (sf[i] > max) {
                max = sf[i];
                sfactor = i;
            }
        }
        return sfactor;
    }
}
