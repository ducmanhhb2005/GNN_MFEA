package IDPC_DU.Multifactorial_Evolutionary_Algorithm.IDPC_DU_EDGES;

import java.util.Random;
import java.util.Scanner;

public class Main {
    public static int REITERATE_TIME = 10000;
    public static int [] LowestCost;
    public static int [] time;
    public static Random random = new Random(4);
    static Scanner sc = new Scanner(System.in);
    public static int inputTask;

    public static void main(String [] args) {
        System.out.print("Input the number of task: ");
        inputTask = sc.nextInt();
        time = new int [inputTask];
        LowestCost = new int [inputTask];
        for (int i = 0; i < LowestCost.length; i++) {
            LowestCost[i] = 99999;
        }

        //Giai doan tien xu ly
        Files.CreateData(inputTask);
        Files.ReadFile(inputTask);
        //Loc canh
        for (int i = 0; i < Files.TASK_NUM; i++) {
            PreProcess.EdgeFilter(Files.edges[i]);
        }

        //Tao khong gian tim kiem dong nhat
        PreProcess.CreateUSS(Files.edges);

        //Tach rieng cac canh cua tung node
        for (int i = 0; i < Files.TASK_NUM; i++) {
            PreProcess.SeperateData(Files.edges[i], Files.edgesTask.get(i));
        }

        //Khoi tao quan the
        Population.PopulationInitialize();
        //Danh gia quan the ban dau
        Population.PopulationIriteirate();

        for (int i = 0; i < REITERATE_TIME; i++) {
            Population.PickingForGenOpt();
            Population.PopulationIriteirate();
            Population.NaturalSelection(Population.selectedForGO);

            for (int j = 0; j < Population.population.size(); j++) {
                for (int j2 = 0; j2 < Files.TASK_NUM; j2++) {
                    if(Population.population.get(j).f_rank[j2] == 1){
                        if (Population.population.get(j).f_cost[j2] < LowestCost[j2]){
                            LowestCost[j2] = Population.population.get(j).f_cost[j2];
                            time[j2] = i;
                            break;
                        }
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

        for (int i = 0; i < LowestCost.length; i++) {
            System.out.print("Lowest Cost Task " + i + ": " + LowestCost[i] + " Time " + i + ": "+ time[i]);
            System.out.println(" ");
        }
    }
}
