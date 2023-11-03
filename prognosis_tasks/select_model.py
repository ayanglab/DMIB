import timm
import sys
sys.path.append("..")
from model.backbones.densenet import densenet121
from model.fusion_concat import concat_densenet121_addfc_addrelu
from model.fusion_attention import attention_densenet121_addfc_addrelu
from model.fusion_transformer import TransformerFusion, TransformerImage
from model.clinical_only import Clinical


def select_model(opt, num_fold):
    print(opt.model_name)
    
    if opt.model_name == 'proposed_crossmodal':
        from model.fusion_crossmodal import concat_densenet_image_clinical_densenet121
        model = concat_densenet_image_clinical_densenet121(pretrained=opt.pretrained, num_classes=2, drop_rate=opt.drop_rate, len_clinical=opt.len_clinical, opt=opt)
    elif opt.model_name == 'concat':
        model = concat_densenet121_addfc_addrelu(num_classes=2, drop_rate=opt.drop_rate, len_clinical=opt.len_clinical, num_fold=num_fold, opt=opt)
    elif opt.model_name == 'attention':
        model = attention_densenet121_addfc_addrelu(num_classes=2, drop_rate=opt.drop_rate, len_clinical=opt.len_clinical, num_fold=num_fold, opt=opt)
    elif opt.model_name == 'transformer':
        model = TransformerFusion(len_clinical=opt.len_clinical)
    # elif opt.model_name == 'TransformerImage':
    #     model = TransformerImage()
    elif opt.model_name == 'dynamic':
        from model.fusion_dynamic import concat_densenet_image_clinical_densenet121
        model = concat_densenet_image_clinical_densenet121(pretrained=opt.pretrained, num_classes=2, drop_rate=opt.drop_rate, len_clinical=opt.len_clinical, opt=opt)
    else:
        model = densenet121(pretrained=opt.pretrained, num_classes=2, drop_rate=opt.drop_rate)
        
    return model
