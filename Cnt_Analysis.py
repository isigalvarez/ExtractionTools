# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 16:55:17 2018

Vamos a juntar aquí las instrucciones para hacer las imágenes para Bilbao

@author: Izzy
"""

import itertools

import ExhalationTools as ExT
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Recalculamos?
flag_update=0
# Ruta hacia los datos


#= Exhalación relativa con expGen =============================================
# Extraemos datos
df_Raw,_ = ExT.retrieve_data(flag_update=flag_update)
# Aplicamos criterios de calidad
idx = (df_Raw['pValue'] >= 0.05) & (df_Raw['R2Adj'] >= 0.4)
# Limpiamos el dataframe
df = df_Raw[idx]

# Calculamos la media de referencia
idx_Cnt = (df['Chamber']=='Cnt')
E_ref,sE_ref = ExT.compute_meanAndError(df[idx_Cnt])
print('\nExhalación de referencia: ' + str(E_ref) + ' +- ' + str(sE_ref))
# Calculamos otras medias
idx_aG = idx_Cnt & (df['Device']=='aG')
idx_aC = idx_Cnt & (df['Device']=='aC')
idx_R7 = idx_Cnt & (df['Device']=='R7Sniff')
E_aG,sE_aG = ExT.compute_meanAndError(df[idx_aG])
E_aC,sE_aC = ExT.compute_meanAndError(df[idx_aC])
E_R7,sE_R7 = ExT.compute_meanAndError(df[idx_R7])
print('\nExhalación aG: ' + str(E_aG) + ' +- ' + str(sE_aG) +
      '\nExhalación aC: ' + str(E_aC) + ' +- ' + str(sE_aC) +
      '\nExhalación R7: ' + str(E_R7) + ' +- ' + str(sE_R7))

# Filtramos df
shared_features=[('Method','CC'),('Soil','Cnt')]    
values_to_avoid = [('Chamber',['V15ne','V35ne','VUPC']),
                   ('Device',['RS'])]
df = ExT.filter_experiments(df,shared_features,values_to_avoid)
# Traducimos equipos para que las etiquetas sean más bonicas
deviceDict = {'R7Sniff':'Rad 7','aG':'Alphaguard 1','aC':'Alphaguard 2'}
df['Device'] = df['Device'].replace(deviceDict)
# Traducimos cámaras para que sean coherentes con el proceeding
chamberDict = {'Cnt':'Reference Box','V10':'V10','V15':'V14','V35':'V30'}
df['Chamber'] = df['Chamber'].replace(chamberDict)

# Cribamos
idx_noCnt =  (df['Chamber']!='Reference Box') & (df['Device']!='RS')
df_noCnt = df[idx_noCnt]
# Calculamos la tasa de exhalación relativa
df_noCnt['E/E_ref'] = df_noCnt['E (Bqm-2h-1)']/E_ref

## Ploteamos la tasa de exhalación relativa
# Creamos una paletta asignado a cada equipo un color
devices_palette = dict(zip(['Alphaguard 1','Alphaguard 2','Rad 7'], sns.color_palette('Set1')))
# Define plotting style
sns.set_style('ticks')
# Create fig and plot barplot
fig1 = plt.figure()
ax1 = sns.barplot(data=df_noCnt,x='Chamber',y='E/E_ref',
                  hue='Device',hue_order=['Alphaguard 1','Alphaguard 2','Rad 7'],
                  palette=devices_palette,capsize=0.05)

#== Añadimos parches al barplot ====
# Para añadir parches a un barplot de seaborn es necesario iterarar sobre el
# atributo parches, 'barplot.patches'.
# Seaborn dibuja primero cada subgrupo antes de pasar al siguiente, es decir,
# dibuja primero las primeras barras de cada grupo, luego las segundas, etc.
# Por tanto, al iterar tendremos que ir teniendo en cuenta si hemos cambiado de
# grupo o no antes de poner el parche.
#
# Vamos a ello:
# Calculamos el numero de subgrupos distintos
num_locations = len(df_noCnt['Chamber'].unique())
# Definimos un generador con patrones
hatches = itertools.cycle(['x', '//', '\\'])
# Itreamos sobre cada barra
for i, bar in enumerate(ax1.patches):
    if i % num_locations == 0:
        hatch = next(hatches)
    bar.set_hatch(hatch)
#===================================

# Añadimos la leyenda                
leg = ax1.legend(frameon=True,framealpha=0.8,loc='lower left',
                 title='Device') 
# Cambiamos la etiqueta
ax1.set_ylabel('$E/E_{ref}$')
# Añadimos lineas horizontales para el 1 y el 10%
ax1.axhline(y=1,color='k',ls='-')
ax1.axhline(y=0.9,color='gray',ls='--')
ax1.axhline(y=1.1,color='gray',ls='--')
# Ajustamos y guardamos
fig1.tight_layout()
fig1.savefig('Cnt_Exhalation_expGenFit.jpg',dpi=500)

#===================================
## Ploteamos la constante de tiempo efectiva
fig2 = plt.figure()
# Plot de boxplot
order = ['Reference Box','V10','V14','V30','C0','C3','C6','C9']
hue_order = ['Alphaguard 1','Alphaguard 2','Rad 7']
ax2 = sns.barplot(x='Chamber',y='Lamb (s-1)',hue='Device',
                 order=order,hue_order=hue_order,
                 data = df,palette=devices_palette,capsize=0.05)
                 
#== Añadimos parches al barplot ====
# Para añadir parches a un boxplot vamos a acceder directamente a las "intancias"
# de cada caja via 'boxplot.artist'.
                 
# Calculamos el numero de subgrupos distintos
# Calculamos el numero de subgrupos distintos
num_locations = len(df['Chamber'].unique())
# Definimos un generador con patrones
hatches = itertools.cycle(['x', '//', '\\'])
# Itreamos sobre cada barra
for i, bar in enumerate(ax2.patches):
    if i % num_locations == 0:
        hatch = next(hatches)
    bar.set_hatch(hatch)
#===================================

# Añadimos la leyenda                
leg = ax2.legend(frameon=True,framealpha=0.8,loc='best',
                 title='Device') 
# Set ylabel
ax2.set_ylabel('$\lambda_{ef}$ $(s^{-1})$')
# Configure grid
ax2.grid(which='major',axis='y',color='k')
ax2.grid(which='minor',axis='y')
# Ponemos escala logaritmica
ax2.set_yscale('log')
# Ponemos el minimo en 1e-6
ax2.set_ylim(bottom=1e-6)
# Añadimos una linea con la constante del radon
ax2.axhline(y=2.1e-6,color='r',ls='--')
# Añadimos una etiqueta
ax2.text(-0.88,2.0e-6,'$\lambda_{Rn}$')
# Ajustamos y guardamos
fig2.tight_layout()
fig2.savefig('Cnt_Lambda_expGenFit.jpg',dpi=500)
#==============================================================================

#== Exhalación relativa para ajuste lineal ====================================
ExT.linearFit_analysis(values_to_avoid=values_to_avoid,
                       flag_update=flag_update)
#==============================================================================

# == Analisis para 2h con expGen ==============================================
# Extremos datos
df2h_Raw, fitErrors_2h = ExT.retrieve_data(t_duration=120,flag_update=flag_update)
# Aplicamos criterios de calidad
idx = (df2h_Raw['pValue'] >= 0.05) & (df2h_Raw['R2Adj'] >= 0.4)
# Limpiamos el dataframe
df2h = df2h_Raw[idx]
# Filtramos df
shared_features=[('Method','CC'),('Soil','Cnt')]    
values_to_avoid = [('Chamber',['C0','C3','C6','C9','C12','C14','V15ne','V35ne','VUPC'])]
df2h = ExT.filter_experiments(df2h,shared_features,values_to_avoid)
# Anotamos la exhalacion de referencia
df2h['E/E_ref'] = df2h['E (Bqm-2h-1)']/E_ref

# Traducimos equipos para que las etiquetas sean más bonicas
deviceDict = {'R7Sniff':'Rad 7','aG':'Alphaguard 1','aC':'Alphaguard 2'}
df2h['Device'] = df2h['Device'].replace(deviceDict)
# Traducimos cámaras para que sean coherentes con el proceeding
chamberDict = {'Cnt':'Reference Box','V15':'V14','V35':'V30'}
df2h['Chamber'] = df2h['Chamber'].replace(chamberDict)

# Ploteamos
devices_palette = dict(zip(['Alphaguard 1','Alphaguard 2','Rad 7'], sns.color_palette('Set1')))
#sdnfasd
fig3 = plt.figure()
ax = sns.barplot(data=df2h,x='Chamber',y='E/E_ref',
                 hue='Device',hue_order=['Alphaguard 1','Alphaguard 2','Rad 7'],
                 palette=devices_palette,capsize=0.05)
# Calculamos el numero de subgrupos distintos
num_locations = len(df2h['Chamber'].unique())
# Definimos un generador con patrones
hatches = itertools.cycle(['x', '//', '\\'])
# Itreamos sobre cada barra
for i, bar in enumerate(ax.patches):
    if i % num_locations == 0:
        hatch = next(hatches)
    bar.set_hatch(hatch)
# Añadimos leyenda
leg = ax.legend(frameon=True,framealpha=0.8,loc='lower left',
                title='Device') 
# Cambiamos la etiqueta
ax.set_ylabel('$E/E_{ref}$')
# Añadimos lineas horizontales para el 1 y el 10%
ax.axhline(y=1,color='k',ls='-')
ax.axhline(y=0.9,color='gray',ls='--')
ax.axhline(y=1.1,color='gray',ls='--')
# Guardamos
fig3.tight_layout()
fig3.savefig('Exh_expGen_10-120.jpeg',dpi=1000)

## Mirramos cosillas del error relativo
# Calculamos el error relativo
df['sE (%)'] = df['sE (Bqm-2h-1)']/df['E (Bqm-2h-1)']*100
df2h['sE (%)'] = df2h['sE (Bqm-2h-1)']/df2h['E (Bqm-2h-1)']*100
# Sacamos cosas

# =============================================================================

# == Analisis para 2h con exp ==============================================
# Extremos datos
df2h_Raw, fitErrors_2h = ExT.retrieve_data(fitType='exp',t_duration=120,flag_update=flag_update)
# Aplicamos criterios de calidad
idx = (df2h_Raw['pValue'] >= 0.05) & (df2h_Raw['R2Adj'] >= 0.4)
# Limpiamos el dataframe
df2h = df2h_Raw[idx]
# Filtramos df
shared_features=[('Method','CC'),('Soil','Cnt')]    
values_to_avoid = [('Chamber',['C0','C3','C6','C9','C12','C14','V15ne','V35ne','VUPC'])]
df2h = ExT.filter_experiments(df2h,shared_features,values_to_avoid)
# Anotamos la exhalacion de referencia
df2h['E/E_ref'] = df2h['E (Bqm-2h-1)']/E_ref

# Traducimos equipos para que las etiquetas sean más bonicas
deviceDict = {'R7Sniff':'Rad 7','aG':'Alphaguard 1','aC':'Alphaguard 2'}
df2h['Device'] = df2h['Device'].replace(deviceDict)
# Traducimos cámaras para que sean coherentes con el proceeding
chamberDict = {'Cnt':'Reference Box','V15':'V14','V35':'V30'}
df2h['Chamber'] = df2h['Chamber'].replace(chamberDict)

# Ploteamos
devices_palette = dict(zip(['Alphaguard 1','Alphaguard 2','Rad 7'], sns.color_palette('Set1')))
#sdnfasd
fig1 = plt.figure()
order = ['Reference Box','V10','V14','V30','C0','C3','C6','C9','C12','C14']
hue_order = ['Alphaguard 1','Alphaguard 2','Rad 7']
ax = sns.barplot(data=df2h,x='Chamber',y='E/E_ref',
                 hue='Device',order=order,hue_order=hue_order,
                 palette=devices_palette,capsize=0.05)
leg = ax.legend(frameon=True,framealpha=0.8,loc='lower left',
                title='Device') 
# Cambiamos la etiqueta
ax.set_ylabel('$E/E_{ref}$')
# Añadimos lineas horizontales para el 1 y el 10%
ax.axhline(y=1,color='k',ls='-')
ax.axhline(y=0.9,color='gray',ls='--')
ax.axhline(y=1.1,color='gray',ls='--')
# Guardamos
fig1.tight_layout()

## Mirramos cosillas del error relativo
# Calculamos el error relativo
df2h['sE (%)'] = df2h['sE (Bqm-2h-1)']/df2h['E (Bqm-2h-1)']*100
# =============================================================================