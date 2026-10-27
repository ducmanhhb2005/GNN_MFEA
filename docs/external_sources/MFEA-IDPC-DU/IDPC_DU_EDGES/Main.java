package IDPC_DU.Multifactorial_Evolutionary_Algorithm.IDPC_DU_EDGES;

import java.util.Random;

public class Main {
    public static int REITERATE_TIME = 10000;
    public static int LowestCost1 = 99999, LowestCost2 = 99999, time1, time2;
    public static Random random = new Random(4);

    public static void main(String[] args) {
        //Giai doan tien xu ly
        Files.ReadFile();
        Files.edges1 = PreProcess.EdgeFilter(Files.edges1);
        Files.edges2 = PreProcess.EdgeFilter(Files.edges2);

        //Tao khong gian tim kiem dong nhat
        PreProcess.CreateUSS(Files.edges1, Files.edges2);

        //Tach rieng cac canh cua tung node
        PreProcess.SeperateData(Files.edges1, Files.edges_1);
        PreProcess.SeperateData(Files.edges2, Files.edges_2);

        //Khoi tao quan the
        Population.PopulationInitialize();
        //Danh gia quan the ban dau
        Population.PopulationIriteirate();

        for (int i = 0; i < REITERATE_TIME; i++) {
            Population.PickingForGenOpt();
            Population.PopulationIriteirate();
            Population.NaturalSelection(Population.selectedForGO);

            for (int j = 0; j < Population.population.size(); j++) {
                if(Population.population.get(j).f_rank[0] == 1){
                    if (Population.population.get(j).f_cost[0] < LowestCost1){
                        LowestCost1 = Population.population.get(j).f_cost[0];
                        time1 = i;
                        break;
                    }
                }
            }
            for (int v = 0; v < Population.population.size(); v++) {
                if(Population.population.get(v).f_rank[1] == 1){
                    if (Population.population.get(v).f_cost[1] < LowestCost2){
                        LowestCost2 = Population.population.get(v).f_cost[1];
                        time2 = i;
                        break;
                    }
                }   
            }
        }
        
        for (Circle c : Population.population) {       
            for (int i = 0; i < c.nodePri.length; i++) {
                System.out.print(c.nodePri[i] + " ");
            }
            System.out.println(" ");
            for (int i = 0; i < c.edgeIdx.length; i++) {
                System.out.print(c.edgeIdx[i] + " ");
            }
            System.out.println(" ");

            for (int i = 0; i < c.f_rank.length; i++) {
                System.out.print(c.f_cost[i]+ " " + c.f_rank[i]+ " " );
            }
            System.out.println(" ");
        }

        System.out.print("Lowest Cost Task 1 " + LowestCost1 + " Time1 " + time1);
        System.out.println(" ");
        System.out.print("Lowest Cost Task 2 " + LowestCost2 + " Time2 " + time2);
        System.out.println(" ");
        System.out.print("popsize " + Population.POP_SIZE);
    }
}
