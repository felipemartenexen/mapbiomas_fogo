// =======================================================
// Script GEE: Exportar áreas (km²) de newCol4 e fireYear
// para as regiões 1–9 e anos de 2000 a 2023.
// Inclui hotspots, máscara de floresta, componentes conectados,
// filtro ≥50 000 m², focalMode e cálculo de newCol4.
// =======================================================

// 1. Lista de regiões (client‑side)
var nRegions = [2,3,4,5,6,7,8,9];

// 2. Lista de anos (client‑side)
var years = [];
for (var y = 2000; y <= 2023; y++) {
  years.push(y);
}

// 3. Carregue imagens fixas (client‑side)
var fireAll = ee.Image(
  'projects/mapbiomas-public/assets/brazil/fire/collection3/' +
  'mapbiomas_fire_collection3_annual_burned_v1'
);
var colecao = ee.Image(
  'projects/mapbiomas-public/assets/brazil/lulc/collection9/' +
  'mapbiomas_collection90_integration_v1'
);

// 4. Array onde vamos acumular todas as features
var features = [];

// 5. Loop por cada região
nRegions.forEach(function(nRegion) {
  // 5.1 monte o region FeatureCollection
  var idRegion = nRegion + 10;
  var region = ee.FeatureCollection(
    "users/geomapeamentoipam/AUXILIAR/regioes_biomas"
  ).filterMetadata('region','equals', idRegion);

  // 5.2 loop por cada ano
  years.forEach(function(year) {
    var yearStr = year.toString();
    var prevStr = (year - 1).toString();

    // 5.2.1 bandas queimadas ano e ano-1
    var fireYear = fireAll.select('burned_area_' + yearStr).clip(region);
    var firePrev = fireAll.select('burned_area_' + prevStr).clip(region);

    // 5.2.2 hotspots (assets externos)
    var hs     = ee.Image(
      "projects/ee-felipe-martenexen/assets/fire_col4/fireAgain/hotspot_" 
      + yearStr
    ).eq(0);
    var hsPrev = ee.Image(
      "projects/ee-felipe-martenexen/assets/fire_col4/fireAgain/hotspot_" 
      + prevStr
    ).eq(0);

    // 5.2.3 máscara de floresta (MapBiomas Col9, classe 3)
    var forest = ee.Image(1).mask(
      colecao.select('classification_' + (year-2).toString()).eq(3)
    ).clip(region);

    // 5.2.4 identifique repetições (fireAgain)
    var fireAgain = fireYear.updateMask(firePrev);

    // 5.2.5 componentes conectados e filtro por área ≥50 000 m²
    var fireAgainId = fireAgain.connectedComponents({
      connectedness: ee.Kernel.square(1),
      maxSize: 128
    });
    var pixelArea = ee.Image.pixelArea();
    var objectArea = fireAgainId.multiply(pixelArea);
    var areaMask = objectArea.gte(100000);
    fireAgainId = fireAgainId.updateMask(areaMask);

    // 5.2.6 aplique hotspots sobre as áreas repetidas
    hs = hs.updateMask(fireAgain);

    // 5.2.7 calcule fireAgainSize (exclui objetos grandes e hotspots)
    var fireAgainSize = fireAgain
      .updateMask(fireAgainId.select('labels').unmask().not());

    // 5.2.8 suavização por modo focal (800 m)
    fireAgainSize = fireAgainSize.updateMask(
      hs.focalMode(800, 'square', 'meters', 1).unmask().not()
    );

    // 5.2.9 fireNoAgain (áreas repetidas sem hotspots, dentro de floresta)
    var fireNoAgain = fireAgain
      .updateMask(fireAgainSize.unmask().not())
      .updateMask(forest);

    // 5.2.10 newCol4 = queimadas deste ano excluindo fireAgainSize
    var newCol4 = fireYear.updateMask(fireNoAgain.unmask().not());

    Export.image.toAsset({
      image: newCol4,
      description: 'col4_r' + nRegion + '_v4_' +  year,
      assetId: 'projects/ee-felipe-martenexen/assets/fire_col4/fireAgain/image/' + 'col4_r' + nRegion + '_v4_' +  year,
      region: region.geometry().bounds(),
      scale: 30,
      maxPixels: 1e13 
      })
    // 5.2.11 calcule áreas em km²
   //var statsFire = fireYear
   //  .gt(0)
   //  .multiply(pixelArea)
   //  .reduceRegion({
   //    reducer:   ee.Reducer.sum(),
   //    geometry:  region,
   //    scale:     30,
   //    maxPixels: 1e13
   //  });
   //var areaFireYear = ee.Number(
   //  statsFire.get('burned_area_' + yearStr)
   //).divide(1e6);

   //var statsNew = newCol4
   //  .gt(0)
   //  .multiply(pixelArea)
   //  .reduceRegion({
   //    reducer:   ee.Reducer.sum(),
   //    geometry:  region,
   //    scale:     30,
   //    maxPixels: 1e13
   //  });
   //var areaNewCol4 = ee.Number(
   //  statsNew.get('burned_area_' + yearStr)
   //).divide(1e6);

    // 5.2.12 empilhe a Feature
    //features.push(ee.Feature(null, {
    //  'nRegion':            nRegion,
    //  'year':               year,
    //  'area_fireYear_km2':  areaFireYear,
    //  'area_newCol4_kja':   areaNewCol4
    //}));
  });
});

// 6. Monte a FeatureCollection final e imprima
//var table = ee.FeatureCollection(features);
//print(table);
//
//// 7. Exporte tudo para o Google Drive em CSV
//Export.table.toDrive({
//  folder:      'EXPORT-GEE',
//  collection:  table,
//  description: 'areas_newCol4_fireYear_regions_2000_2023',
//  fileFormat:  'CSV'
//});
