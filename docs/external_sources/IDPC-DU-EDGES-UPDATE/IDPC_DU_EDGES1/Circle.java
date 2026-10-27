package IDPC_DU.Multifactorial_Evolutionary_Algorithm.IDPC_DU_EDGES1;

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

    public Circle(int [] sampleNode, int [] nodeOut) {
        int pos, tmp;
        this.nodePri = new int[sampleNode.length];
        this.edgeIdx = new int[sampleNode.length];

        for (int i = 0; i < sampleNode.length; i++) {
            nodePri[i] = sampleNode[i];
        }

        //Set the priority for each edge
        for (int i = 0; i < nodePri.length; i++) {
            pos = Mfea4IDPCDU.random.nextInt(nodePri.length - 1);
            tmp = nodePri[i];
            nodePri[i] = nodePri[pos];
            nodePri[pos] = tmp;
        }

        //Set the edge-out index for each edge
        for (int i = 0; i < edgeIdx.length; i++) {
            if (nodeOut[i] == 0) {
                edgeIdx[i] = -1;
            } else {
                pos = Mfea4IDPCDU.random.nextInt(nodeOut[i]);
                edgeIdx[i] = nodeOut[i];
            }
        }
    }

    public int getTaskFcost(int task) {
        return f_cost[task];
    }

    public double getBestFit() {
        return best_fit;
    }

    public void CalculateFactorialCost(int task) {
        int cost = 0, maxPri = 0, des, nextNode = -1, isVisited;
        List<Integer> adjacentNode = new ArrayList<Integer>();

        this.domainLeft.clear();
        this.nodeLeft.clear();

        isVisited = Files.SRC[task];
        des = Files.DES[task];
        this.nodeLeft.add(isVisited);
        int [] tmpNP = new int [Files.NODES_NUM[task]];
        int [] tmpEI = new int [Files.NODES_NUM[task]];

        if (Files.NODES_NUM[task] < this.nodePri.length){
            tmpNP = DecodeNodePri(this.nodePri, Files.NODES_NUM[task]);
            tmpEI = DecodeEdgeIdx(this.edgeIdx, Files.NODES_NUM[task]);
        } else {
            tmpNP = this.nodePri;
            tmpEI = this.edgeIdx;
        }

        while (isVisited != des) {
            //  System.out.print("visit"  +  isVisited);
            //  System.out.println(" ");
            //Check which node is the adjacent one
            for (int j = 0; j < Files.NODES_NUM[task]; j++) {
                for (int i = 0; i < Files.edgesTask.get(task)[isVisited-1][j].size(); i++) {
                    if (Files.edgesTask.get(task)[isVisited-1][j].get(i).src == isVisited
                    && !adjacentNode.contains(Files.edgesTask.get(task)[isVisited-1][j].get(i).des)) {
                        adjacentNode.add(Files.edgesTask.get(task)[isVisited-1][j].get(i).des);   
                        //System.out.print(Files.edgesTask.get(task)[isVisited-1][j].get(i).des + " ");
                    }
                }   
            }
            //System.out.println(" ");
            if (adjacentNode.size() == 0) {
                cost += 9999999;
                break;
            } 
            
            //Check which one got the highest priority based on adjacent nodes
            for (int i = 0; i < adjacentNode.size(); i++) {
                if (maxPri < tmpNP[adjacentNode.get(i)-1] && !this.nodeLeft.contains(adjacentNode.get(i))) {
                    maxPri = tmpNP[adjacentNode.get(i)-1];
                    nextNode = adjacentNode.get(i);
                }
            }
                
            //Calculate the factorial cost
            int penalty = 99999;
            if (this.domainLeft.contains(Files.edgesTask.get(task)[isVisited-1][nextNode-1].get(this.edgeIdx[isVisited-1] % Files.edgesTask.get(task)[isVisited-1][nextNode-1].size()).domain)) {
                cost += penalty;
            } else {
                cost += Files.edgesTask.get(task)[isVisited-1][nextNode-1].get(tmpEI[isVisited-1]  % Files.edgesTask.get(task)[isVisited-1][nextNode-1].size()).weight;
                this.domainLeft.add(Files.edgesTask.get(task)[isVisited-1][nextNode-1].get(tmpEI[isVisited-1] % Files.edgesTask.get(task)[isVisited-1][nextNode-1].size()).domain);
            }
            isVisited = nextNode;
            this.nodeLeft.add(isVisited);
            maxPri = 0;
            adjacentNode.clear();
        }
        //System.out.println("end while");
        //System.out.println(" ");
        this.f_cost[Files.TASK[task]] = cost; 
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

    public static Comparator<Circle> TaskCostComp = new Comparator<Circle>(){

        public int compare(Circle c1, Circle c2) {
            int cost1 = c1.getTaskFcost(Population.curTask);
            int cost2 = c2.getTaskFcost(Population.curTask);

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