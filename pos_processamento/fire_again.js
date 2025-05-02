// =======================================================
// Script GEE: Exportar áreas (km²) de newCol4 e fireYear
// para as regiões 1–9 e anos de 2000 a 2023.
// Inclui hotspots, máscara de floresta, componentes conectados,
// filtro ≥50 000 m², focalMode e cálculo de newCol4.
// =======================================================

// 1. Lista de regiões (client‑side)
var nRegions = [2,3];

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
    var firePrevius = fireAll.select('burned_area_' + prevStr).clip(region);

    // 5.2.2 hotspots (assets externos)
    var annualHotSpotsImg = ee.Image("projects/ee-felipe-martenexen/assets/fire_col4/fireAgain/hotspot_" + yearStr).eq(0);
    //
    var annualHotSpotsPreviusImg = ee.Image("projects/ee-felipe-martenexen/assets/fire_col4/fireAgain/hotspot_" + prevStr).eq(0);

    // 5.2.3 máscara de floresta (MapBiomas Col9, classe 3)
    var forest = ee.Image(1).mask(
      colecao.select('classification_' + (yearStr-2).toString()).eq(3)
    );

    // 5.2.4 identifique repetições (fireAgain)
    var fireAgain = fireYear.updateMask(firePrevius);

    annualHotSpotsImg = annualHotSpotsImg.updateMask(fireAgain);
    
     // Defina o raio em metros (500 metros)
    var radius = 1000;
    
    // Defina o tipo de kernel ("circle" para uma janela circular)
    var kernelType = 'square';
    
    // Defina a unidade como metros
    var units = 'meters';
    
    // O número de iterações (você pode definir mais se quiser suavizar mais a imagem)
    var iterations = 1;
    
    // Aplique o foco da moda (focal_mode)
    var fireAgainFocalMode = annualHotSpotsImg.focalMode(radius, kernelType, units, iterations);
    
    //fireAgainSize = fireAgainSize.updateMask(fireAgainFocalMode.unmask().not());
    
    var fireNoAgain = fireAgain.updateMask(fireAgainFocalMode.unmask().not());
    
    fireNoAgain = fireNoAgain.updateMask(forest);
    
    var fireAgainId = fireNoAgain.connectedComponents({
      connectedness: ee.Kernel.square(1),
      maxSize: 128
    });
    
    // Compute the number of pixels in each object defined by the "labels" band.
    var objectSize = fireAgainId.select('labels')
      .connectedPixelCount({
        maxSize: 128, eightConnected: true
      });
      
    var pixelArea = ee.Image.pixelArea();
    
    var objectArea = objectSize.multiply(pixelArea);
    
    var areaMask = objectArea.lt(50000);
    
    var fireAgainIdLt = fireAgainId.updateMask(areaMask).select('labels');
    
    var fireNoAgainSize = fireNoAgain.updateMask(fireAgainIdLt.unmask().not());
    
    var newCol4 = fireYear.updateMask(fireNoAgainSize.unmask().not());

    newCol4 = ee.Image(1).mask(newCol4);
    
    newCol4 = newCol4.rename('b1').unmask().float();
    
    Export.image.toAsset({
      image: newCol4,
      description: 'col4_r' + nRegion + '_v5_' +  year,
      assetId: 'projects/ee-felipemartenexen/assets/fire_col4/fireAgain/image/' + 'col4_r' + nRegion + '_v5_' +  year,
      region: region.geometry().bounds(),
      scale: 30,
      maxPixels: 1e13 
    });

  });
});

