# 3D_datagen

## Clone this repo
'''bash
git clone https://github.com/VaclavKnapp/3D_datagen.git
'''

## Infinigen
1. Clone the infinigen repo
   '''bash
   git clone https://github.com/princeton-vl/infinigen.git
   '''
2. move the code from the 3D_datagen/infigen folder in the infinigen repo folder
   '''bash
   mv 3D_datagen/infinigen/gen_assests_blend.py infinigen/infinigen_examples
   mv 3D_datagen/infinigen/oddity.py infinigen/.
   mv 3D_datagen/infinigen/render.sh infinigen/.
   '''
### Generating objects
1. Generating .blend files
   '''bash
   cd infinigen
   '''
   '''bash
   python infinigen_examples/gen_assests_blend.py -f 
