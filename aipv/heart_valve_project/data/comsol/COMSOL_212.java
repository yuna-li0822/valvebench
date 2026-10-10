/*
 * COMSOL_212.java
 */

import com.comsol.model.*;
import com.comsol.model.util.*;

/** Model exported on Oct 2 2026, 11:46 by COMSOL 6.4.0.429. */
public class COMSOL_212 {

  public static Model run() {
    Model model = ModelUtil.create("Model");

    model.modelPath("C:\\Users\\yiyef\\Downloads");

    model.label("2.12COMSOL.mph");

    model.param().set("U0", "5[mm]");
    model.param().set("alpha", "0");
    model.param().set("dU", "0.02[mm]");
    model.param().set("tr", "10[ms]");
    model.param().set("ton", "50[ms]");
    model.param().set("tr2", "20[ms]");
    model.param().set("toff", "150[ms]");
    model.param().set("currentiter", "0");

    model.component().create("comp1", true);

    model.component("comp1").geom().create("geom1", 3);

    model.result().table().create("evl3", "Table");
    model.result().table().create("tbl1", "Table");
    model.result().table().create("tbl2", "Table");
    model.result().table().create("tbl3", "Table");
    model.result().table().create("tbl4", "Table");
    model.result().table().create("tbl5", "Table");
    model.result().table().create("tbl6", "Table");
    model.result().table().create("tbl7", "Table");
    model.result().table().create("tbl8", "Table");
    model.result().table().create("tbl9", "Table");

    model.func().create("step1", "Step");
    model.func("step1").active(false);
    model.func("step1").set("location", 0.05);
    model.func("step1").set("smooth", 0.9);

    model.component("comp1").mesh().create("mesh1");

    model.component("comp1").geom("geom1").geomRep("cadps");
    model.component("comp1").geom("geom1").designBooleans(false);
    model.component("comp1").geom("geom1").create("imp1", "Import");
    model.component("comp1").geom("geom1").feature("imp1").set("type", "cad");
    model.component("comp1").geom("geom1").feature("imp1")
         .set("filename", "C:\\Users\\yueyu\\Downloads\\Assembly (1).step");
    model.component("comp1").geom("geom1").feature("imp1").importData();
    model.component("comp1").geom("geom1").run();

    model.component("comp1").material().create("mat1", "Common");
    model.component("comp1").material().create("mat2", "Common");
    model.component("comp1").material("mat1").selection().set();
    model.component("comp1").material("mat1").info().create("Composition");
    model.component("comp1").material("mat1").propertyGroup()
         .create("LossTangentDF", "LossTangentDF", "Loss tangent, dissipation factor");
    model.component("comp1").material("mat1").propertyGroup("LossTangentDF").func()
         .create("RPT_0_to_50_relative_humidity_tested_at_23C_1", "Piecewise");
    model.component("comp1").material("mat1").selection().set(2, 3);
    model.component("comp1").material("mat2").selection().set(1);
    model.component("comp1").material("mat2").info().create("Composition");
    model.component("comp1").material("mat2").propertyGroup("def").func().create("rho", "Piecewise");

    model.component("comp1").coordSystem().create("sys2", "Cylindrical");

    model.component("comp1").physics().create("solid", "SolidMechanics", "geom1");
    model.component("comp1").physics("solid").create("fix1", "Fixed", 2);
    model.component("comp1").physics("solid").create("sma1", "ShapeMemoryAlloy", 3);
    model.component("comp1").physics("solid").feature("sma1").selection().set(1);
    model.component("comp1").physics("solid").create("disp1", "Displacement2", 2);
    model.component("comp1").physics("solid").feature("disp1").selection().set(15, 18);

    model.component("comp1").mesh("mesh1").autoMeshSize(1);

    model.result().table("evl3").label("Evaluation 3D");
    model.result().table("evl3").comments("\u4ea4\u4e92\u7684\u4e09\u7ef4\u503c");
    model.result().table("tbl1").label("\u8868\u683c 1");
    model.result().table("tbl1").comments("\u4f53\u79ef\u5e73\u5747 1");
    model.result().table("tbl2").label("\u8868\u683c 2");
    model.result().table("tbl2").comments("\u4f53\u79ef\u5e73\u5747 1");
    model.result().table("tbl3").label("\u8868\u683c 3");
    model.result().table("tbl3").comments("\u4f53\u79ef\u5206 1");
    model.result().table("tbl4").label("\u8868\u683c 4");
    model.result().table("tbl4").comments("\u4f53\u79ef\u6d4b\u91cf 1");
    model.result().table("tbl5").label("\u8868\u683c 5");
    model.result().table("tbl5").comments("\u4f53\u79ef\u5e73\u5747 1");
    model.result().table("tbl6").label("\u8868\u683c 6");
    model.result().table("tbl6").comments("\u8868\u9762\u79ef\u5206 1");
    model.result().table("tbl7").label("\u8868\u683c 7");
    model.result().table("tbl7").comments("\u4f53\u6700\u5927\u503c 1");
    model.result().table("tbl8").label("\u8868\u683c 8");
    model.result().table("tbl8").comments("\u4f53\u79ef\u5206 1");
    model.result().table("tbl9").label("\u8868\u683c 9");
    model.result().table("tbl9").comments("\u4f53\u79ef\u5206 1");

    model.component("comp1").material("mat1")
         .label("Acetal copolymer [0% to 50% relative humidity,tested at 23C (296K)]");
    model.component("comp1").material("mat1")
         .set("info", new String[][]{{"Composition", "", "acetal POM polyoxymethylene"}});
    model.component("comp1").material("mat1").propertyGroup("LossTangentDF")
         .func("RPT_0_to_50_relative_humidity_tested_at_23C_1").label("Piecewise");
    model.component("comp1").material("mat1").propertyGroup("LossTangentDF")
         .func("RPT_0_to_50_relative_humidity_tested_at_23C_1").set("arg", "f");
    model.component("comp1").material("mat1").propertyGroup("LossTangentDF")
         .func("RPT_0_to_50_relative_humidity_tested_at_23C_1")
         .set("pieces", new String[][]{{"100.0", "345265.0", "3.71080000E+00"}, {"345265.0", "781456.0", "2.26466954E-01*log(f)^4 - 1.16492162E+01*log(f)^3 + 2.24774882E+02*log(f)^2 - 1.92812223E+03*log(f) + 6.20756121E+03"}, {"781456.0", "1000000.0", "1.56587163E+01*log(f)^3 - 6.38874330E+02*log(f)^2 + 8.68914504E+03*log(f) - 3.93911378E+04"}});
    model.component("comp1").material("mat1").propertyGroup("LossTangentDF")
         .func("RPT_0_to_50_relative_humidity_tested_at_23C_1").set("argunit", "1/s");
    model.component("comp1").material("mat1").propertyGroup("LossTangentDF")
         .set("epsilonPrim", new String[]{"RPT_0_to_50_relative_humidity_tested_at_23C_1(freq)", "0", "0", "0", "RPT_0_to_50_relative_humidity_tested_at_23C_1(freq)", "0", "0", "0", "RPT_0_to_50_relative_humidity_tested_at_23C_1(freq)"});
    model.component("comp1").material("mat1").propertyGroup("LossTangentDF")
         .setPropertyInfo("epsilonPrim", "Reference: Celcon POM, Acetal Copolymer (POM), Designing with Celcon, POM-046_Celcon_DesignGuideTG_AM_0913.pdf available online at https://www.celanese.com/-/media/Engineered%20Materials/Files/Product%20Technical%20Guides/POM-046_Celcon_DesignGuideTG_AM_0913.pdf\nNote: tested at 23C (296K), 0% to 50% relative humidity, measurements on 1 mm (0.039 inch) sheet in accordance with ASTM D 150-61\nReference temperature: 296.15[K]");
    model.component("comp1").material("mat1").propertyGroup("LossTangentDF").addInput("frequency");
    model.component("comp1").material("mat2").label("PVC [solid]");
    model.component("comp1").material("mat2").set("info", new String[][]{{"Composition", "", "(C2H3Cl)n"}});
    model.component("comp1").material("mat2").propertyGroup("def").func("rho").set("arg", "T");
    model.component("comp1").material("mat2").propertyGroup("def").func("rho")
         .set("pieces", new String[][]{{"293.0", "303.0", "1350.0"}});
    model.component("comp1").material("mat2").propertyGroup("def").func("rho").set("argunit", "K");
    model.component("comp1").material("mat2").propertyGroup("def").func("rho").set("fununit", "kg/m^3");
    model.component("comp1").material("mat2").propertyGroup("def").set("density", "rho(T)");
    model.component("comp1").material("mat2").propertyGroup("def")
         .setPropertyInfo("density", "Reference: room temperature value\nNote: Tg near 82C (355K)");
    model.component("comp1").material("mat2").propertyGroup("def").addInput("temperature");

    model.component("comp1").physics("solid").prop("ShapeProperty").set("order_displacement", 1);
    model.component("comp1").physics("solid").prop("StructuralTransientBehavior")
         .set("StructuralTransientBehavior", "Quasistatic");
    model.component("comp1").physics("solid").prop("AdvancedSettings").set("GroupPhysOdesRc", false);
    model.component("comp1").physics("solid").prop("AdvancedSettings").set("GroupPhysOdesAtt", false);
    model.component("comp1").physics("solid").prop("AdvancedSettings").set("GroupPhysOdesRd", false);
    model.component("comp1").physics("solid").feature("lemm1").set("E_mat", "userdef");
    model.component("comp1").physics("solid").feature("lemm1").set("E", "2.7*10^9");
    model.component("comp1").physics("solid").feature("lemm1").set("nu_mat", "userdef");
    model.component("comp1").physics("solid").feature("lemm1").set("nu", 0.35);
    model.component("comp1").physics("solid").feature("lemm1").set("rho_mat", "userdef");
    model.component("comp1").physics("solid").feature("lemm1").set("rho", 1410);
    model.component("comp1").physics("solid").feature("dcnt1")
         .set("pn", "((min(1e-3*(5^niterCMP),1))*solid.dcnt1.E_char)/solid.hmin_dst");
    model.component("comp1").physics("solid").feature("fix1").active(false);
    model.component("comp1").physics("solid").feature("sma1").set("TransformationFunction", "Cosine");
    model.component("comp1").physics("solid").feature("sma1")
         .set("df_F", "(((solid.rho*solid.sma1.bM)*solid.xi)+(solid.sma1.mu1+solid.sma1.mu2))+(solid.sma1.sY*(solid.sma1.ramp2(solid.xi)))");
    model.component("comp1").physics("solid").feature("sma1")
         .set("df_R", "(((solid.rho*solid.sma1.bA)*solid.xi)+(solid.sma1.mu1-solid.sma1.mu2))-(solid.sma1.sY*(solid.sma1.ramp2(1-solid.xi)))");
    model.component("comp1").physics("solid").feature("sma1").set("T0_mat", "userdef");
    model.component("comp1").physics("solid").feature("sma1").set("T0", "313.15[K]");
    model.component("comp1").physics("solid").feature("sma1").set("nu_mat", "userdef");
    model.component("comp1").physics("solid").feature("sma1").set("nu", 0.33);
    model.component("comp1").physics("solid").feature("sma1").set("AusteniteMaterial", "mat2");
    model.component("comp1").physics("solid").feature("sma1").set("E_A_mat", "userdef");
    model.component("comp1").physics("solid").feature("sma1").set("E_A", "7.0*10^10");
    model.component("comp1").physics("solid").feature("sma1").set("Cp_A_mat", "userdef");
    model.component("comp1").physics("solid").feature("sma1").set("Cp_A", 837);
    model.component("comp1").physics("solid").feature("sma1").set("MartensiteMaterial", "mat2");
    model.component("comp1").physics("solid").feature("sma1").set("E_M_mat", "userdef");
    model.component("comp1").physics("solid").feature("sma1").set("E_M", "3.0*10^10");
    model.component("comp1").physics("solid").feature("sma1").set("Cp_M_mat", "userdef");
    model.component("comp1").physics("solid").feature("sma1").set("Cp_M", 837);
    model.component("comp1").physics("solid").feature("sma1").set("TMs_mat", "userdef");
    model.component("comp1").physics("solid").feature("sma1").set("TMs", "283.15[K]");
    model.component("comp1").physics("solid").feature("sma1").set("TMf_mat", "userdef");
    model.component("comp1").physics("solid").feature("sma1").set("TMf", "273.15[K]");
    model.component("comp1").physics("solid").feature("sma1").set("CM_mat", "userdef");
    model.component("comp1").physics("solid").feature("sma1").set("CM", "6.0*10^6");
    model.component("comp1").physics("solid").feature("sma1").set("TAs_mat", "userdef");
    model.component("comp1").physics("solid").feature("sma1").set("TAs", "303.15[K]");
    model.component("comp1").physics("solid").feature("sma1").set("TAf_mat", "userdef");
    model.component("comp1").physics("solid").feature("sma1").set("TAf", "313.15[K]");
    model.component("comp1").physics("solid").feature("sma1").set("CA_mat", "userdef");
    model.component("comp1").physics("solid").feature("sma1").set("CA", "7.0*10^6");
    model.component("comp1").physics("solid").feature("sma1").set("etrmaxLagoudas_mat", "userdef");
    model.component("comp1").physics("solid").feature("sma1").set("etrmaxLagoudas", 0.04);
    model.component("comp1").physics("solid").feature("sma1").set("sigmaStar_mat", "userdef");
    model.component("comp1").physics("solid").feature("sma1").set("sigmaStar", "2*10^8");
    model.component("comp1").physics("solid").feature("sma1").set("rho_mat", "userdef");
    model.component("comp1").physics("solid").feature("sma1").set("rho", 6450);
    model.component("comp1").physics("solid").feature("sma1")
         .set("minput_strainreferencetemperature_src", "userdef");
    model.component("comp1").physics("solid").feature("sma1").set("minput_strainreferencetemperature", "298.15[K]");
    model.component("comp1").physics("solid").feature("sma1").set("minput_temperature_src", "userdef");
    model.component("comp1").physics("solid").feature("sma1").set("minput_temperature", "310.15[K]");
    model.component("comp1").physics("solid").feature("sma1").feature("trdir1")
         .set("trdir", "flc2hs(1[s]-t, 0.05[s]) - flc2hs(t-1[s], 0.05[s])");
    model.component("comp1").physics("solid").feature("disp1")
         .set("Direction", new String[][]{{"prescribed"}, {"prescribed"}, {"prescribed"}});
    model.component("comp1").physics("solid").feature("disp1")
         .set("U0", new String[][]{{"-0.005[m] * ( (t/1[s])*(1-flc2hs(t-1[s],0.05[s])) + ((2[s]-t)/1[s])*flc2hs(t-1[s],0.05[s])*(1-flc2hs(t-2[s],0.05[s])))"}, {"0"}, {"0"}});
    model.component("comp1").physics("solid").feature("disp1").set("coordinateSystem", "sys2");

    model.study().create("std8");
    model.study("std8").create("time", "Transient");

    model.sol().create("sol8");
    model.sol("sol8").attach("std8");

    model.result().param().set("dset1_solid_sma1_pt", "1", "\u7528\u4e8e\u76f8\u56fe\u7684\u70b9");
    model.result().param().set("dset1_solid_sma1_Tmax", "350[K]", "\u76f8\u56fe\u7684\u6700\u9ad8\u6e29\u5ea6");
    model.result().param().set("dset3_solid_sma1_pt", "1", "\u7528\u4e8e\u76f8\u56fe\u7684\u70b9");
    model.result().param().set("dset3_solid_sma1_Tmax", "350[K]", "\u76f8\u56fe\u7684\u6700\u9ad8\u6e29\u5ea6");
    model.result().param().set("dset4_solid_sma1_pt", "1", "\u7528\u4e8e\u76f8\u56fe\u7684\u70b9");
    model.result().param().set("dset4_solid_sma1_Tmax", "350[K]", "\u76f8\u56fe\u7684\u6700\u9ad8\u6e29\u5ea6");
    model.result().param().set("dset6_solid_sma1_pt", "1", "\u7528\u4e8e\u76f8\u56fe\u7684\u70b9");
    model.result().param().set("dset6_solid_sma1_Tmax", "350[K]", "\u76f8\u56fe\u7684\u6700\u9ad8\u6e29\u5ea6");
    model.result().param().set("dset7_solid_sma1_pt", "1", "\u7528\u4e8e\u76f8\u56fe\u7684\u70b9");
    model.result().param().set("dset7_solid_sma1_Tmax", "350[K]", "\u76f8\u56fe\u7684\u6700\u9ad8\u6e29\u5ea6");
    model.result().param().set("dset8_solid_sma1_pt", "1", "\u7528\u4e8e\u76f8\u56fe\u7684\u70b9");
    model.result().param().set("dset8_solid_sma1_Tmax", "350[K]", "\u76f8\u56fe\u7684\u6700\u9ad8\u6e29\u5ea6");
    model.result().param().set("dset9_solid_sma1_pt", "1", "\u7528\u4e8e\u76f8\u56fe\u7684\u70b9");
    model.result().param().set("dset9_solid_sma1_Tmax", "350[K]", "\u76f8\u56fe\u7684\u6700\u9ad8\u6e29\u5ea6");
    model.result().param().set("dset10_solid_sma1_pt", "1", "\u7528\u4e8e\u76f8\u56fe\u7684\u70b9");
    model.result().param().set("dset10_solid_sma1_Tmax", "350[K]", "\u76f8\u56fe\u7684\u6700\u9ad8\u6e29\u5ea6");
    model.result().param().set("dset11_solid_sma1_pt", "1", "\u7528\u4e8e\u76f8\u56fe\u7684\u70b9");
    model.result().param().set("dset11_solid_sma1_Tmax", "350[K]", "\u76f8\u56fe\u7684\u6700\u9ad8\u6e29\u5ea6");
    model.result().param().set("dset12_solid_sma1_pt", "1", "\u7528\u4e8e\u76f8\u56fe\u7684\u70b9");
    model.result().param().set("dset12_solid_sma1_Tmax", "350[K]", "\u76f8\u56fe\u7684\u6700\u9ad8\u6e29\u5ea6");
    model.result().param().set("dset13_solid_sma1_pt", "1", "\u7528\u4e8e\u76f8\u56fe\u7684\u70b9");
    model.result().param().set("dset13_solid_sma1_Tmax", "350[K]", "\u76f8\u56fe\u7684\u6700\u9ad8\u6e29\u5ea6");
    model.result().dataset().create("dset8", "Solution");
    model.result().dataset().remove("dset1");
    model.result().numerical().create("max1", "MaxVolume");
    model.result().numerical("max1").selection().set(1);

    model.study("std8").label("1");
    model.study("std8").feature("time").label("\u77ac\u6001");
    model.study("std8").feature("time").set("tlist", "range(0, 0.01, 0.2)");

    model.sol("sol8").createAutoSequence("std8");
    model.sol("sol8").label("\u89e3 8");

    model.study("std8").runNoGen();

    model.result().numerical("max1").label("\u4f53\u6700\u5927\u503c 1");
    model.result().numerical("max1").set("looplevelinput", new String[]{"manual"});
    model.result().numerical("max1").set("table", "tbl7");
    model.result().numerical("max1").set("expr", new String[]{"solid.misesGp"});
    model.result().numerical("max1").set("unit", new String[]{"N/m^2"});
    model.result().numerical("max1").set("descr", new String[]{"von Mises \u5e94\u529b"});
    model.result().numerical("max1")
         .set("const", new String[][]{{"solid.refpntx", "0", "\u529b\u77e9\u8ba1\u7b97\u53c2\u8003\u70b9\uff0cx \u5750\u6807"}, {"solid.refpnty", "0", "\u529b\u77e9\u8ba1\u7b97\u53c2\u8003\u70b9\uff0cy \u5750\u6807"}, {"solid.refpntz", "0", "\u529b\u77e9\u8ba1\u7b97\u53c2\u8003\u70b9\uff0cz \u5750\u6807"}});
    model.result().numerical("max1").setResult();

    return model;
  }

  public static void main(String[] args) {
    run();
  }

}
