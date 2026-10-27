package IDPC_DU.Multifactorial_Evolutionary_Algorithm.IDPC_DU_EDGES;

public class GeneticMechanism {
    public static float ratio = 0.1f;
    public static float rmp = 0.5f;
    public static int point, num;

    public static void GeneticOperators (Circle c1, Circle c2) {
        float rand = Main.random.nextInt(10) * 0.1f;
        Circle child1 = new Circle(c1); 
        Circle child2 = new Circle(c2);

        if((c1.s_factor == c2.s_factor ) || (rand < rmp)) {   //Crossover Condition
            int a, b;
            a = Main.random.nextInt(c1.nodePri.length - 1) + 1;
            b = Main.random.nextInt(c1.nodePri.length - 1) + 1;   
            if (a > b) {
                num = a - b + 1;
                point = b;
            }
            else {
                point = a;
                num = b - a + 1;
            }

            //Knowledge transfer via crossover
            for (int i = 0; i < c1.nodePri.length; i++) {
                if (i >= point && i < num + point) {
                    child1.nodePri[i] = c1.nodePri[i];
                    child2.nodePri[i] = c2.nodePri[i];
                }
                else {
                    child1.nodePri[i] = -1;
                    child2.nodePri[i] = -1;
                }
            }
        
            //PMX Crossover + TwoPointsCrossover
            PMXCrossOver(c1.nodePri, child2.nodePri);
            PMXCrossOver(c2.nodePri, child1.nodePri);
            TwoPointsCrossover(c1.edgeIdx, child2.edgeIdx, point, num);
            TwoPointsCrossover(c2.edgeIdx, child1.edgeIdx, point, num);

            if(c1.s_factor == c2.s_factor ) {                                   //Parents have the same skill factor
                child1.s_factor = c1.s_factor;
                child2.s_factor = c1.s_factor;
            }
            else {                                                              //Parents have different skill factors
                rand = Main.random.nextInt(10) * 0.1f;
                if (rand > 0.5f) {
                    child1.s_factor = c1.s_factor;
                    child2.s_factor = c1.s_factor;
                } 
                else {
                    child1.s_factor = c2.s_factor;
                    child2.s_factor = c2.s_factor;
                }
            }
        } 
        else {                      
            for (int i = 0; i < c1.nodePri.length; i++) {
                child1.nodePri[i] = c1.nodePri[i];
                child2.nodePri[i] = c2.nodePri[i];
            }                                           
            
            child1.s_factor = c1.s_factor;
            child2.s_factor = c2.s_factor;

            Mutation(child1);
            Mutation(child2);
        }

        child1.f_cost[Files.TASK1] = child1.CalculateFactorialCost(child1, true);
        child1.f_cost[Files.TASK2] = child1.CalculateFactorialCost(child1, false);
        child2.f_cost[Files.TASK1] = child2.CalculateFactorialCost(child2, true);
        child2.f_cost[Files.TASK2] = child2.CalculateFactorialCost(child2, false);
        Population.population.add(child1);
        Population.population.add(child2);
    }
    
    public static void PMXCrossOver(int[] parent, int[] child) {
        int k, pos;
        for (int i = 0; i < child.length; i++) {
            k = parent[i];
            if (child[i] == -1) {
                pos = checkValue(child, k);

                if (pos == -1) {
                    child[i] = k;
                }
                else {
                    while(pos != -1) {
                        k = parent[pos];
                        pos = checkValue(child, k);
                    }
                    child[i] = k;
                }
            }
        }
    }

    public static int checkValue(int[] child, int value) {
        int pos = -1;
        for (int i = 0; i < child.length; i++) {
            if (child[i] == value) {
                pos = i;
            }
        }
        return pos;
    }

    public static void TwoPointsCrossover(int[] parent, int[] child, int pos, int num) {
        for (int i = pos; i < pos + num; i++) {
            child[i] = parent[i];
        }
    }

    public static void Mutation(Circle c) {
        int pos1, pos2, tmp;
        
        //Switch node priority
        pos1 = Main.random.nextInt(c.nodePri.length);
        pos2 = Main.random.nextInt(c.nodePri.length);
        tmp = c.nodePri[pos1];
        c.nodePri[pos1] = c.nodePri[pos2];
        c.nodePri[pos2] = tmp;

        //Randomly change one edge index
        pos1 = Main.random.nextInt(c.edgeIdx.length);
        c.edgeIdx[pos1] = Main.random.nextInt(PreProcess.node_out_edge.length -1) + 1;
    }
}
