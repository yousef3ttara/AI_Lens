"""
AI Lens - Plant Leaf Disease Information & Agronomic Recommendations
Provides curated botanical metadata, symptom descriptions, and management
strategies for all 38 classes of the PlantVillage dataset across 14 crop species.
"""

from typing import Dict, Any

DISEASE_DATABASE: Dict[str, Dict[str, Any]] = {
    "Apple___Apple_scab": {
        "species": "Apple (Malus domestica)",
        "condition": "Apple Scab",
        "is_healthy": False,
        "pathogen": "Venturia inaequalis (Ascomycete fungus)",
        "severity": "Moderate to High",
        "symptoms": (
            "Dull olive-green to black velvety spots on leaf surfaces. Leaves may become distorted, "
            "puckered, and drop prematurely, causing tree defoliation and fruit lesions."
        ),
        "cultural_control": (
            "Rake and destroy fallen leaves in autumn to disrupt the fungal overwintering cycle. "
            "Prune canopy regularly to promote sunlight penetration and air movement."
        ),
        "treatment": (
            "Apply approved preventative fungicides (e.g., Captan, Mancozeb, or copper-based sprays) "
            "during early spring bud break through petal fall."
        ),
        "preventative_care": (
            "Plant resistant apple varieties (e.g., Liberty, Enterprise, Honeycrisp). "
            "Avoid overhead irrigation to keep foliage dry."
        ),
    },
    "Apple___Black_rot": {
        "species": "Apple (Malus domestica)",
        "condition": "Black Rot (Frogeye Leaf Spot)",
        "is_healthy": False,
        "pathogen": "Botryosphaeria obtusa (Fungal pathogen)",
        "severity": "High",
        "symptoms": (
            "Small purple spots expand into 'frogeye' lesions with light tan centers and purple margins. "
            "Twigs may develop cankers, and fruit develops firm, concentric rotting zones."
        ),
        "cultural_control": (
            "Prune out dead wood, fire-blight strikes, and mummified fruits where spores overwinter. "
            "Burn or bury infected material away from the orchard."
        ),
        "treatment": (
            "Apply protective fungicides (Captan, thiophanate-methyl) starting from pink bud stage "
            "through cover sprays as recommended by local agronomy guidelines."
        ),
        "preventative_care": (
            "Maintain overall orchard sanitation, prune out cankers during dry winter weather, "
            "and protect trees from mechanical injury and insect wounding."
        ),
    },
    "Apple___Cedar_apple_rust": {
        "species": "Apple (Malus domestica)",
        "condition": "Cedar Apple Rust",
        "is_healthy": False,
        "pathogen": "Gymnosporangium juniperi-virginianae (Heteroecious rust fungus)",
        "severity": "Moderate",
        "symptoms": (
            "Bright yellow-orange spots on upper leaf surfaces that enlarge and develop black pycnia. "
            "Underneath the leaf, raised tube-like fungal aecia emerge late spring/early summer."
        ),
        "cultural_control": (
            "Remove nearby Eastern red cedar (Juniperus virginiana) and ornamental junipers within "
            "a 1 to 2 mile radius when feasible, or prune galls before spore release."
        ),
        "treatment": (
            "Apply rust-effective fungicides (Myclobutanil, Mancozeb) at pink-bud stage until "
            "cedar gall tendrils dry up."
        ),
        "preventative_care": (
            "Select rust-resistant apple cultivars such as Freedom, Liberty, or William's Pride. "
            "Monitor nearby cedar trees for gelatinous spore horns in wet spring weather."
        ),
    },
    "Apple___healthy": {
        "species": "Apple (Malus domestica)",
        "condition": "Healthy Foliage",
        "is_healthy": True,
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Leaf tissue exhibits uniform green pigmentation, smooth margins, and intact vascular structures.",
        "cultural_control": "Continue standard seasonal canopy management, balanced soil nutrition, and clean orchard floor.",
        "treatment": "No remedial chemical treatments necessary. Continue routine integrated pest management (IPM).",
        "preventative_care": "Ensure adequate soil moisture and perform routine annual pruning during dormancy.",
    },
    "Blueberry___healthy": {
        "species": "Blueberry (Vaccinium spp.)",
        "condition": "Healthy Foliage",
        "is_healthy": True,
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Vibrant, glossy green leaves with no signs of chlorosis, necrosis, rust, or powdery mildew.",
        "cultural_control": "Maintain acidic soil pH (4.5 to 5.2) using elemental sulfur or peat moss.",
        "treatment": "No chemical application needed. Monitor soil pH and moisture balance.",
        "preventative_care": "Apply pine bark mulch to preserve soil acidity, retain root moisture, and suppress weeds.",
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "species": "Cherry (Prunus avium / Prunus cerasus)",
        "condition": "Powdery Mildew",
        "is_healthy": False,
        "pathogen": "Podosphaera clandestina (Erysiphaceae fungus)",
        "severity": "Moderate",
        "symptoms": (
            "White, talcum-powder-like fungal patches on young foliage and shoots. Leaves become "
            "curled, crinkled, and brittle; shoot growth may be stunted."
        ),
        "cultural_control": (
            "Prune inner orchard canopy to maximize air movement. Avoid excessive high-nitrogen "
            "fertilization which stimulates tender, susceptible tissue."
        ),
        "treatment": (
            "Apply sulfur-based fungicides, potassium bicarbonate, or horticultural oils at the "
            "first sign of fungal colonies."
        ),
        "preventative_care": (
            "Irrigate via drip systems rather than overhead sprinklers. Regularly inspect new terminal shoots."
        ),
    },
    "Cherry_(including_sour)___healthy": {
        "species": "Cherry (Prunus avium / Prunus cerasus)",
        "condition": "Healthy Foliage",
        "is_healthy": True,
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Foliage is lush, dark green, and free from leaf spots, shot-holes, or fungal powder.",
        "cultural_control": "Maintain regular pruning schedule to facilitate good airflow throughout the canopy.",
        "treatment": "No chemical intervention required.",
        "preventative_care": "Apply dormant copper sprays in late fall to prevent bacterial canker infections.",
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "species": "Corn / Maize (Zea mays)",
        "condition": "Gray Leaf Spot (Cercospora Leaf Spot)",
        "is_healthy": False,
        "pathogen": "Cercospora zeae-maydis (Fungus)",
        "severity": "High",
        "symptoms": (
            "Narrow, rectangular lesions running parallel to leaf veins, initially tan/brown and "
            "turning grayish as spores mature. Blighting leads to premature senescence."
        ),
        "cultural_control": (
            "Rotate crops annually with non-host species (such as soybeans). Practice deep tillage "
            "or crop residue management to break down fungal debris."
        ),
        "treatment": (
            "Apply triazole, strobilurin, or premixed fungicides when disease pressure is noted "
            "prior to or at tassel stage (VT-R1)."
        ),
        "preventative_care": (
            "Plant corn hybrids with high genetic tolerance or resistance to gray leaf spot."
        ),
    },
    "Corn_(maize)___Common_rust_": {
        "species": "Corn / Maize (Zea mays)",
        "condition": "Common Rust",
        "is_healthy": False,
        "pathogen": "Puccinia sorghi (Basidiomycete fungus)",
        "severity": "Moderate",
        "symptoms": (
            "Small, oval to elongate cinnamon-brown pustules scattered across both leaf surfaces. "
            "Pustules rupture epidermal tissue to release powdery urediniospores."
        ),
        "cultural_control": (
            "Plant resistant varieties. Early planting can help avoid peak windborne spore migration."
        ),
        "treatment": (
            "Fungicide treatment (e.g., azoxystrobin or propiconazole) is only economical when rust "
            "pustules appear before tasseling on susceptible hybrid lines."
        ),
        "preventative_care": (
            "Monitor regional rust spore advisories and select resistant corn hybrids."
        ),
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "species": "Corn / Maize (Zea mays)",
        "condition": "Northern Corn Leaf Blight (NCLB)",
        "is_healthy": False,
        "pathogen": "Exserohilum turcicum (Fungus)",
        "severity": "High",
        "symptoms": (
            "Large, elongated, cigar-shaped grayish-green to tan lesions (1 to 6 inches long). "
            "Dark sporulation occurs in damp weather, severely reducing photosynthetic leaf area."
        ),
        "cultural_control": (
            "Rotate fields with non-grass crops and incorporate corn stubble to speed fungal decomposition."
        ),
        "treatment": (
            "Apply foliar fungicides during vegetative growth up to tasseling if weather favors disease."
        ),
        "preventative_care": (
            "Choose resistant hybrids with single-gene (Ht) or multigenic quantitative resistance."
        ),
    },
    "Corn_(maize)___healthy": {
        "species": "Corn / Maize (Zea mays)",
        "condition": "Healthy Foliage",
        "is_healthy": True,
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Uniform deep green leaf blades with clear venation and no necrosis or rust pustules.",
        "cultural_control": "Ensure balanced nitrogen and micronutrient supply according to soil testing.",
        "treatment": "No pesticide intervention needed.",
        "preventative_care": "Maintain proper crop spacing and weed management throughout the vegetative period.",
    },
    "Grape___Black_rot": {
        "species": "Grape (Vitis vinifera / labrusca)",
        "condition": "Black Rot",
        "is_healthy": False,
        "pathogen": "Phyllosticta ampelicida (Guignardia bidwellii)",
        "severity": "High",
        "symptoms": (
            "Small, circular reddish-brown leaf spots with distinct dark margins containing tiny black pycnidia. "
            "Berries shrivel into hard, black, wrinkled mummies."
        ),
        "cultural_control": (
            "Prune out mummified berry clusters, dead canes, and infected foliage during winter. "
            "Ensure vine training maximizes canopy airflow."
        ),
        "treatment": (
            "Apply protective fungicides (Mancozeb, Captan, Myclobutanil) starting from early shoot growth (1-3 inches) "
            "through 4-5 weeks post-bloom."
        ),
        "preventative_care": (
            "Keep vineyard free of wild grapevines and clean vineyard floor of debris."
        ),
    },
    "Grape___Esca_(Black_Measles)": {
        "species": "Grape (Vitis vinifera / labrusca)",
        "condition": "Esca (Black Measles / Trunk Disease)",
        "is_healthy": False,
        "pathogen": "Complex of vascular fungi (Phaeomoniella chlamydospora, Phaeoacremonium)",
        "severity": "Critical",
        "symptoms": (
            "'Tiger-stripe' chlorotic and necrotic interveinal patterns on leaves. Berries develop "
            "small, dark purple spots resembling measles, and vine can suffer sudden apoplexy."
        ),
        "cultural_control": (
            "Avoid pruning wounds in damp weather. Protect large wood cuts with wound sealant or "
            "Trichoderma-based biocontrol pastes."
        ),
        "treatment": (
            "No chemical cure exists for deep trunk vascular infections. Severely affected vines "
            "should be trunk-renewed or carefully eradicated."
        ),
        "preventative_care": (
            "Practice double-pruning or delayed late-winter pruning to reduce infection risk through cuts."
        ),
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "species": "Grape (Vitis vinifera / labrusca)",
        "condition": "Leaf Blight (Pseudocercospora vitis / Isariopsis)",
        "is_healthy": False,
        "pathogen": "Pseudocercospora vitis (Fungus)",
        "severity": "Moderate",
        "symptoms": (
            "Irregular brown necrotic lesions on leaves, often bounded by larger veins, with dense "
            "olive-brown velvety mold under high humidity."
        ),
        "cultural_control": (
            "Thin out grapevine foliage to improve sunlight exposure and speed leaf drying.",
        ),
        "treatment": (
            "Apply broad-spectrum copper fungicides or mancozeb as part of late-season canopy protection."
        ),
        "preventative_care": (
            "Sanitize vineyard debris after harvest and maintain balanced vine vigor."
        ),
    },
    "Grape___healthy": {
        "species": "Grape (Vitis vinifera / labrusca)",
        "condition": "Healthy Foliage",
        "is_healthy": True,
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Well-expanded green leaves with intact lobed margins and healthy green shoots.",
        "cultural_control": "Continue standard shoot positioning, suckering, and leaf pulling.",
        "treatment": "No curative treatments needed.",
        "preventative_care": "Regular scouting for mildew and insect vector populations.",
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "species": "Citrus / Orange (Citrus sinensis)",
        "condition": "Citrus Greening (Huanglongbing / HLB)",
        "is_healthy": False,
        "pathogen": "Candidatus Liberibacter asiaticus (Bacterium, vectored by Asian citrus psyllid)",
        "severity": "Critical",
        "symptoms": (
            "Asymmetrical blotchy mottle on leaves, yellow shoot shoots, vein corking, stunted tree "
            "canopy, and bitter, lopsided fruit that stays green at base."
        ),
        "cultural_control": (
            "Strict vector control of the Asian citrus psyllid (Diaphorina citri) using insecticides "
            "and biological predators (Tamarixia radiata)."
        ),
        "treatment": (
            "HLB has no cure. Remove and destroy infected trees to prevent grove-wide spread. "
            "Nutritional foliar sprays can temporarily support declining trees."
        ),
        "preventative_care": (
            "Plant certified disease-free nursery trees. Monitor orchards using yellow sticky traps."
        ),
    },
    "Peach___Bacterial_spot": {
        "species": "Peach (Prunus persica)",
        "condition": "Bacterial Spot",
        "is_healthy": False,
        "pathogen": "Xanthomonas arboricola pv. pruni (Bacterium)",
        "severity": "High",
        "symptoms": (
            "Water-soaked angular purple-brown spots on leaf blades, which drop out leaving a 'shot-hole' "
            "appearance. Leaves yellow and defoliate; fruit develops pitting and gumming."
        ),
        "cultural_control": (
            "Avoid orchard sites on excessively light sandy soils without windbreaks. Prune canopies for air flow."
        ),
        "treatment": (
            "Apply dormant copper sprays in late fall and early spring. Low-rate copper or oxytetracycline "
            "sprays may be used during the growing season according to local regulations."
        ),
        "preventative_care": (
            "Plant resistant peach varieties (e.g., Redhaven, Candor, Harrow Beauty). Avoid excessive nitrogen."
        ),
    },
    "Peach___healthy": {
        "species": "Peach (Prunus persica)",
        "condition": "Healthy Foliage",
        "is_healthy": True,
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Glossy lanceolate leaves with clear serrated edges and vibrant green coloring.",
        "cultural_control": "Maintain open center pruning and regular soil moisture.",
        "treatment": "No chemical application necessary.",
        "preventative_care": "Apply protective dormant oil and copper sprays prior to bud swelling.",
    },
    "Pepper,_bell___Bacterial_spot": {
        "species": "Bell Pepper (Capsicum annuum)",
        "condition": "Bacterial Spot",
        "is_healthy": False,
        "pathogen": "Xanthomonas euvesicatoria / perforans (Bacterium)",
        "severity": "High",
        "symptoms": (
            "Small, water-soaked, circular to irregular lesions that become dark brown with yellow halos. "
            "Severe leaf drop can occur, leading to sunscald on developing peppers."
        ),
        "cultural_control": (
            "Use certified disease-free seeds and transplants. Avoid overhead irrigation and working in wet fields."
        ),
        "treatment": (
            "Apply fixed copper mixed with mancozeb, or actigard (systemic acquired resistance inducer) "
            "early in the infection cycle."
        ),
        "preventative_care": (
            "Practice 2-3 year crop rotation away from solanaceous plants (tomatoes, peppers, potatoes)."
        ),
    },
    "Pepper,_bell___healthy": {
        "species": "Bell Pepper (Capsicum annuum)",
        "condition": "Healthy Foliage",
        "is_healthy": True,
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Healthy dark green, smooth leaves with intact margins and vigorous upright vegetative growth.",
        "cultural_control": "Ensure consistent drip irrigation and balanced potassium and calcium nutrition.",
        "treatment": "No remediation required.",
        "preventative_care": "Mulch soil surface with plastic or organic mulch to prevent splash contamination.",
    },
    "Potato___Early_blight": {
        "species": "Potato (Solanum tuberosum)",
        "condition": "Early Blight",
        "is_healthy": False,
        "pathogen": "Alternaria solani (Fungus)",
        "severity": "Moderate to High",
        "symptoms": (
            "Dark brown to black necrotic spots with characteristic concentric rings ('bullseye' or target pattern) "
            "on older lower foliage, progressing upwards."
        ),
        "cultural_control": (
            "Maintain optimal plant nutrition; plants stressed by low nitrogen or drought are highly susceptible. "
            "Remove diseased haulms at harvest."
        ),
        "treatment": (
            "Apply preventative protectant fungicides (Chlorothalonil, Mancozeb) alternated with site-specific "
            "fungicides (Boscalid, Azoxystrobin) to prevent resistance."
        ),
        "preventative_care": (
            "Rotate potato fields with non-solanaceous crops on a minimum 3-year cycle. Use drip irrigation."
        ),
    },
    "Potato___Late_blight": {
        "species": "Potato (Solanum tuberosum)",
        "condition": "Late Blight",
        "is_healthy": False,
        "pathogen": "Phytophthora infestans (Oomycete pathogen)",
        "severity": "Critical",
        "symptoms": (
            "Pale green, water-soaked spots rapidly enlarging into large purplish-black blighted areas. "
            "Under humid conditions, white fluffy mycelium appears on leaf undersides, collapsing entire vines in days."
        ),
        "cultural_control": (
            "Destroy all volunteer potatoes and cull piles. Eliminate infected plants immediately upon discovery "
            "to prevent regional airborne spore dispersal."
        ),
        "treatment": (
            "Apply targeted oomycete fungicides (e.g., fluopicolide, mandipropamid, cymoxanil, chlorothalonil) "
            "immediately upon weather alerts."
        ),
        "preventative_care": (
            "Plant certified certified seed tubers. Monitor local late blight forecasting tools."
        ),
    },
    "Potato___healthy": {
        "species": "Potato (Solanum tuberosum)",
        "condition": "Healthy Foliage",
        "is_healthy": True,
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Robust green composite foliage without lesions, water-soaking, or wilting.",
        "cultural_control": "Hill soil properly around developing tubers to guard against greening and pests.",
        "treatment": "No chemical application needed.",
        "preventative_care": "Maintain preventative scouting routine during cool, wet periods.",
    },
    "Raspberry___healthy": {
        "species": "Raspberry (Rubus idaeus)",
        "condition": "Healthy Foliage",
        "is_healthy": True,
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Bright green leaves with serrated leaflets, free from yellowing, spot necrosis, or powdery mildew.",
        "cultural_control": "Prune out old floricanes after fruiting; maintain trellis support for air flow.",
        "treatment": "No curative treatment required.",
        "preventative_care": "Ensure well-drained soil to protect sensitive root systems from Phytophthora root rot.",
    },
    "Soybean___healthy": {
        "species": "Soybean (Glycine max)",
        "condition": "Healthy Foliage",
        "is_healthy": True,
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Trifoliate leaves exhibiting uniform green coloration without rust pustules or bacterial blighting.",
        "cultural_control": "Ensure proper seedbed preparation and maintain balanced phosphorus/potassium levels.",
        "treatment": "No fungicide or bactericide treatment needed.",
        "preventative_care": "Plant high-germination certified seed inoculated with Bradyrhizobium japonicum.",
    },
    "Squash___Powdery_mildew": {
        "species": "Squash / Cucurbits (Cucurbita spp.)",
        "condition": "Powdery Mildew",
        "is_healthy": False,
        "pathogen": "Podosphaera xanthii / Golovinomyces cichoracearum (Fungi)",
        "severity": "Moderate to High",
        "symptoms": (
            "Powdery white fungal colonies spreading over upper and lower leaf surfaces, petioles, and stems. "
            "Infected leaves turn yellow, brown, and become parchment-like and dry."
        ),
        "cultural_control": (
            "Space squash plants widely to allow maximum sunlight and air drying. Remove older infected foliage."
        ),
        "treatment": (
            "Apply potassium bicarbonate, neem oil, sulfur sprays, or specialized fungicides (Quintec, Rally) "
            "at the very earliest sign of powdery colonies."
        ),
        "preventative_care": (
            "Select powdery mildew-resistant squash and pumpkin cultivars."
        ),
    },
    "Strawberry___Leaf_scorch": {
        "species": "Strawberry (Fragaria × ananassa)",
        "condition": "Leaf Scorch",
        "is_healthy": False,
        "pathogen": "Diplocarpon earlianum (Fungus)",
        "severity": "Moderate",
        "symptoms": (
            "Irregular purplish to dark brown blotches resembling burns on leaves. As spots multiply, "
            "the leaf margins curl upward and the entire leaf turns brown and scorched."
        ),
        "cultural_control": (
            "Renovate strawberry beds after harvest, mowing down and incorporating or removing diseased foliage. "
            "Avoid overhead sprinkling."
        ),
        "treatment": (
            "Apply preventative fungicides (Captan, Thiophanate-methyl) in early spring before blossoms open."
        ),
        "preventative_care": (
            "Use certified disease-free strawberry crowns. Establish rows with adequate spacing for rapid drying."
        ),
    },
    "Strawberry___healthy": {
        "species": "Strawberry (Fragaria × ananassa)",
        "condition": "Healthy Foliage",
        "is_healthy": True,
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Lush trifoliate leaves, rich emerald green with distinct serrations and vigorous runner development.",
        "cultural_control": "Keep plants mulched with clean straw to keep leaves and fruit clean from soil splash.",
        "treatment": "No treatment required.",
        "preventative_care": "Provide consistent moisture via drip irrigation and monitor for spider mites.",
    },
    "Tomato___Bacterial_spot": {
        "species": "Tomato (Solanum lycopersicum)",
        "condition": "Bacterial Spot",
        "is_healthy": False,
        "pathogen": "Xanthomonas perforans / euvesicatoria (Bacterium)",
        "severity": "High",
        "symptoms": (
            "Small (less than 3mm), dark brown water-soaked angular spots on leaves, surrounded by yellow halos. "
            "Foliage eventually scorches and drops; fruit displays small scabby lesions."
        ),
        "cultural_control": (
            "Discard diseased seedlings. Sterilize stakes and greenhouse trays. Never work in wet tomato rows."
        ),
        "treatment": (
            "Spray with copper hydroxide combined with mancozeb, or use bio-bactericides like Bacillus subtilis."
        ),
        "preventative_care": (
            "Rotate crops for at least 2 years away from solanaceous crops. Use certified path-tested seed."
        ),
    },
    "Tomato___Early_blight": {
        "species": "Tomato (Solanum lycopersicum)",
        "condition": "Early Blight",
        "is_healthy": False,
        "pathogen": "Alternaria solani (Fungus)",
        "severity": "Moderate to High",
        "symptoms": (
            "Concentric rings producing a target-board pattern on older lower leaves, surrounded by yellow tissue. "
            "Causes defoliation starting from the ground up, exposing tomatoes to sunscald."
        ),
        "cultural_control": (
            "Prune off the lowest 12-18 inches of leaves once plants are established. Apply thick straw mulch to "
            "stop spores splashing up from the soil."
        ),
        "treatment": (
            "Apply protective fungicides (Chlorothalonil, Copper sprays, or Mancozeb) on a 7-10 day schedule in wet weather."
        ),
        "preventative_care": (
            "Stake or cage tomato plants for upright growth and drip irrigate exclusively at the base."
        ),
    },
    "Tomato___Late_blight": {
        "species": "Tomato (Solanum lycopersicum)",
        "condition": "Late Blight",
        "is_healthy": False,
        "pathogen": "Phytophthora infestans (Oomycete pathogen)",
        "severity": "Critical",
        "symptoms": (
            "Large, irregularly shaped, water-soaked oily spots turning greenish-black. White fungal fuzz appears "
            "underneath in humid conditions; entire plant collapses rapidly."
        ),
        "cultural_control": (
            "Promptly bag and remove infected tomato plants in the trash; do not compost. Keep plants well-ventilated."
        ),
        "treatment": (
            "Apply preventative anti-oomycete sprays (Chlorothalonil, Copper, Mandipropamid) before disease onset."
        ),
        "preventative_care": (
            "Select late-blight resistant cultivars (e.g., Defiant, Mountain Merit, Iron Lady). Avoid planting near potatoes."
        ),
    },
    "Tomato___Leaf_Mold": {
        "species": "Tomato (Solanum lycopersicum)",
        "condition": "Leaf Mold",
        "is_healthy": False,
        "pathogen": "Passalora fulva (Cladosporium fulvum, Fungus)",
        "severity": "Moderate",
        "symptoms": (
            "Pale green to yellow spots on upper leaf surface, with dense olive-brown velvety mold directly underneath. "
            "Most severe in high humidity environments such as high tunnels and greenhouses."
        ),
        "cultural_control": (
            "Increase greenhouse ventilation, reduce relative humidity below 85%, and increase plant spacing."
        ),
        "treatment": (
            "Apply bio-fungicides or copper hydroxide to protect uninfected leaves."
        ),
        "preventative_care": (
            "Prune lower suckers to maximize airflow. Choose resistant hybrid varieties with Cf resistance genes."
        ),
    },
    "Tomato___Septoria_leaf_spot": {
        "species": "Tomato (Solanum lycopersicum)",
        "condition": "Septoria Leaf Spot",
        "is_healthy": False,
        "pathogen": "Septoria lycopersici (Fungus)",
        "severity": "High",
        "symptoms": (
            "Numerous small, circular spots (1-3mm) with dark brown margins and sunken gray or tan centers containing "
            "tiny black specks (pycnidia). Leaves turn yellow and drop prematurely."
        ),
        "cultural_control": (
            "Remove lower infected leaves immediately. Clean up and bury or burn all crop debris in fall."
        ),
        "treatment": (
            "Apply Chlorothalonil or Copper fungicides as soon as lower leaf spotting appears."
        ),
        "preventative_care": (
            "Mulch heavily around tomato bases to prevent rain splash. Practice 3-year crop rotation."
        ),
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "species": "Tomato (Solanum lycopersicum)",
        "condition": "Two-Spotted Spider Mite Infestation",
        "is_healthy": False,
        "pathogen": "Tetranychus urticae (Acari: Tetranychidae pest)",
        "severity": "High",
        "symptoms": (
            "Fine yellow stippling or bronzing on leaves, accompanied by delicate silk webbing on undersides "
            "and leaf axils. Foliage dries, yellows, and drops in hot, dusty weather."
        ),
        "cultural_control": (
            "Rinse foliage with strong jets of water to disrupt mite webs and dislodge colonies. Keep soil consistently moist."
        ),
        "treatment": (
            "Apply insecticidal soaps, horticultural oils, neem oil, or predatory mites (Phytoseiulus persimilis). "
            "Avoid broad-spectrum synthetic pyrethroids which kill beneficial predatory mites."
        ),
        "preventative_care": (
            "Control surrounding weeds which serve as alternate mite hosts. Avoid drought stress."
        ),
    },
    "Tomato___Target_Spot": {
        "species": "Tomato (Solanum lycopersicum)",
        "condition": "Target Spot",
        "is_healthy": False,
        "pathogen": "Corynespora cassiicola (Fungus)",
        "severity": "Moderate to High",
        "symptoms": (
            "Small brown lesions with light brown centers and faint concentric rings. Leaf spots enlarge and coalesce, "
            "leading to severe defoliation."
        ),
        "cultural_control": (
            "Improve air movement within the crop row. Ensure adequate plant spacing and stake tomatoes."
        ),
        "treatment": (
            "Apply fungicides labeled for target spot (e.g., Azoxystrobin, Chlorothalonil, or Boscalid)."
        ),
        "preventative_care": (
            "Avoid planting adjacent to older infected cucurbit, pepper, or tomato plantings."
        ),
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "species": "Tomato (Solanum lycopersicum)",
        "condition": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "is_healthy": False,
        "pathogen": "Begomovirus (Vectored by Bemisia tabaci whiteflies)",
        "severity": "Critical",
        "symptoms": (
            "Severe leaf curling (upward cupping), reduced leaflet size, interveinal chlorosis (yellowing), "
            "and bushy, stunted plant growth with blossom abortion."
        ),
        "cultural_control": (
            "Eradicate infected plants immediately. Install fine insect mesh (50-mesh) on greenhouse vents. "
            "Use reflective silver mulches to repel vector whiteflies."
        ),
        "treatment": (
            "Viruses cannot be cured once inside plant tissue. Control whitefly vectors using insecticidal soaps, "
            "yellow sticky cards, or systemic neonicotinoid/diamide insecticides as legally permitted."
        ),
        "preventative_care": (
            "Plant TYLCV-resistant tomato cultivars (e.g., Tygress, Skyway, Charger, Red Bounty)."
        ),
    },
    "Tomato___Tomato_mosaic_virus": {
        "species": "Tomato (Solanum lycopersicum)",
        "condition": "Tomato Mosaic Virus (ToMV)",
        "is_healthy": False,
        "pathogen": "Tobamovirus (Mechanically transmitted virus)",
        "severity": "High",
        "symptoms": (
            "Mottling of leaves with alternating light and dark green areas, leaf distortion ('shoestringing' or fern-like appearance), "
            "and stunted growth. Fruit may ripen unevenly with internal browning."
        ),
        "cultural_control": (
            "Wash hands thoroughly with soap or 20% nonfat dry milk solution before handling plants. "
            "Smokers must never handle plants without sanitizing (tobacco carries mosaic viruses). Disinfect tools."
        ),
        "treatment": (
            "No chemical cure exists. Immediately remove and incinerate or trash infected plants."
        ),
        "preventative_care": (
            "Purchase certified virus-free seeds and plant TMV/ToMV resistant cultivars (marked with 'T' or 'ToMV')."
        ),
    },
    "Tomato___healthy": {
        "species": "Tomato (Solanum lycopersicum)",
        "condition": "Healthy Foliage",
        "is_healthy": True,
        "pathogen": "None detected",
        "severity": "None",
        "symptoms": "Vibrant green compound leaves without spots, curling, discoloration, or webbing.",
        "cultural_control": "Prune excess suckers, stake properly, and maintain regular watering to prevent blossom end rot.",
        "treatment": "No remedial application necessary.",
        "preventative_care": "Mulch soil around the base and scout weekly for early signs of disease.",
    },
}


def get_disease_info(class_name: str) -> Dict[str, Any]:
    """
    Retrieve botanical metadata, diagnosis, and agronomic recommendations
    for a given PlantVillage class name.
    """
    if class_name in DISEASE_DATABASE:
        return DISEASE_DATABASE[class_name]

    # Graceful fallback for unexpected or unmapped classes
    parts = class_name.split("___")
    species = parts[0].replace("_", " ") if len(parts) > 0 else "Unknown Plant"
    condition_raw = parts[1].replace("_", " ") if len(parts) > 1 else class_name
    is_healthy = "healthy" in class_name.lower()

    return {
        "species": species,
        "condition": condition_raw.title(),
        "is_healthy": is_healthy,
        "pathogen": "Unknown / Unclassified" if not is_healthy else "None",
        "severity": "Low" if is_healthy else "Unspecified",
        "symptoms": "Leaf features detected matching classifier pattern for " + condition_raw,
        "cultural_control": "Ensure general crop sanitation and healthy soil conditions.",
        "treatment": "Consult a certified agricultural extension specialist for a definitive in-field diagnosis.",
        "preventative_care": "Regularly inspect foliage and avoid extreme environmental stresses.",
    }
